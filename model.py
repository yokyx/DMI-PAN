from torchvision.models.video import r3d_18
import torch
import torch.nn as nn
from einops import rearrange
import math

class CustomR3D18(nn.Module):
    def __init__(self):
        super(CustomR3D18, self).__init__()
        self.model = r3d_18(weights='KINETICS400_V1')
        self.model.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)


class DMIPAN(nn.Module):
    """On top of adding positional encoding, add pre-trained network discrimination.

    Args:
        args
    """

    def __init__(self, args):
        super(DMIPAN, self).__init__()

        self.args = args
        self.batch_size = self.args.batch_size
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.bag_size = self.args.num_frames // self.args.instance_length
        self.instance_length = self.args.instance_length
        self.device = self.args.device

        # Backbone networks
        model = r3d_18(weights='KINETICS400_V1')
        self.features = nn.Sequential(
            *list(model.children())[:-1])  # After avgpool 512x1

        # Load model object
        checkpoint = torch.load('/home/dell/yx/ubuntu6/weight/Classifier_pre_train.pth',
                                map_location=self.device)

        # Instantiate the model object
        self.pre_model = CustomR3D18().to(self.device)

        # Load state dictionary
        self.pre_model.load_state_dict(checkpoint['model_state_dict'])

        # Set the model to evaluation mode
        self.pre_model.eval()

        self.lstm = nn.LSTM(input_size=512, hidden_size=512,
                            num_layers=2, batch_first=True, bidirectional=False)

        # Multi-head self-attention
        self.heads = 4
        self.dim_head = 512 // self.heads
        self.scale = self.dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(
            512, (self.dim_head * self.heads) * 3, bias=False)

        self.pwconv = nn.Conv1d(self.instance_length, 1, 3, 1, 1)

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1))

    def Specific_learning(self, fake_label, ins_feture):
        """
        Learn different video processing strategies based on fake labels.
        ① Do not process when both the first and last instance predictions are greater than 3.
        ② If the first instance's prediction is 0, subtract the maximum predicted instance with value 0 from all instances.
        :param x:
        :return:
        """
        batch_size, il_size, _ = fake_label.shape
        for b in range(batch_size):

            first_value = fake_label[b, 0, :]
            last_value = fake_label[b, -1, :]

            if first_value >= 2 and last_value >= 2:
                mean_value = ins_feture[b].clone() .mean(dim=0)
                ins_feture[b, 0, :] = mean_value

            else:
                min_index = torch.argmax(fake_label[b, :, 0])
                min_value = ins_feture[b, min_index, :]

                # Subtract the minimum value from all other values
                ins_feture[b] = ins_feture[b] - min_value

                # Replace min_value with the mean across the instance length
                mean_value = ins_feture[b].clone().mean(dim=0)
                ins_feture[b, min_index, :] = mean_value

        return ins_feture

    def add_positional_encoding(self, x, d_model, max_len, device):
        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        x_with_pe = x + pe[:x.size(0), :]
        return x_with_pe


    def MIL(self, x):
        """The Multi Instance Learning Aggregation of instances

        Inputs:
            x: [batch, bag_size, 512]
        """
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)

        # [batch, bag_size, 1024]
        ori_x = x

        # MHSA
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        x = torch.matmul(attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        # x = self.norm(x)
        x = torch.sigmoid(x)

        x = ori_x * x

        return x


    def decision(self, ins_x, seq_x):

        # Calculate the maximum values and corresponding indices along the last dimension (dim=2) of the ins_x tensor
        max_values, max_indices = torch.max(ins_x, dim=2)

        # Count the number of times the maximum value corresponds to an index of 1 in each batch
        exists_one = (max_indices == 1).sum(dim=1)

        ins_max = torch.zeros((ins_x.shape[0], 2))

        # If there are any batches where there are two or more occurrences of max value indices being 1
        if (exists_one >= 2).any():
            # Iterate over each batch
            for i in range(ins_x.shape[0]):

                if exists_one[i] >= 2:
                    mask = max_indices[i] == 1
                    # Find the maximum value at index 1 in max_values
                    max_val = max_values[i][mask].max()

                    max_val_index = (max_values[i] == max_val) & mask

                    # If there are multiple occurrences of the same maximum value (i.e., multiple True values in max_val_index), randomly select one True index
                    if max_val_index.sum() > 1:
                        true_indices = torch.nonzero(max_val_index).squeeze(1)
                        random_index = true_indices[torch.randint(0, len(true_indices), (1,)).item()]
                    else:
                        random_index = torch.nonzero(max_val_index)[0].item()

                    ins_max[i] = ins_x[i, random_index, :]

            return ins_max

        else:
            return seq_x



    def forward(self, x):
        # x=[batch, num_frames, 3, 224, 224]

        # Handle instances with self.instance_length
        org_x = rearrange(x, 'b (t1 t2) c h w -> (b t1) c t2 h w',
                          t1=self.instance_length, t2=self.bag_size)
        # [batch*il, 3, bag_size, 224, 224]

        ins_x = self.features(org_x).squeeze()
        # [batch*il, 512]

        ins_x = rearrange(ins_x, '(b t) c -> b t c', t=self.instance_length)
        # [batch, il, 512]

        fake_label = self.pre_model(org_x)
        # [batch*il, 1]

        fake_label = rearrange(fake_label, '(b t) c -> b t c', t=self.instance_length)
        # [batch, il, 1]

        # Add differential learning
        ins_x = self.Specific_learning(fake_label, ins_x)
        # [batch, il, 512]

        ins_x = self.add_positional_encoding(ins_x, 512, self.instance_length, self.device)

        # Add attention mechanism in multi-instance learning
        ins_x = self.MIL(ins_x)
        # [batch, il, 512]


        # Direct decision——————————————————————-——
        x = self.pwconv(ins_x).squeeze()
        # [batch, 512]
        out = self.fc(x)
        # [batch, 1]


        # Joint decision——————————————————————-——
        ## Instance aggregation sequence operation
        seq_x = self.pwconv(ins_x).squeeze()
        # [batch, 512]
        seq_x = self.fc(seq_x)
        seq_x = self.softmax(seq_x)
        # [batch, 2]

        ins_x = self.fc(ins_x)
        ins_x = self.softmax(ins_x)
        # [batch*il, 2]

        ins_x = ins_x.view(self.batch_size, self.instance_length, -1)
        # [batch, il, 2]

        out = self.decision(ins_x, seq_x)

        return out
