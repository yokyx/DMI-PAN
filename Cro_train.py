import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from Seq_AveSamp import MyDataSet
from utils import setup_logging_and_dirs, train_one_epoch, evaluate ,plot_confusion_matrix,select_Pred_error
import pandas as pd
from model import DMIPAN
from sklearn.model_selection import GroupKFold
import numpy as np
import logging


# Load the Excel file containing the dataset paths
df = pd.read_excel(r'/home/dell/yx/ubuntu6/excel/UNBC_VAS.xlsx')
images_path = df['Path']
images_label = df['VAS']

seed=42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def main(args):
    # Set device to CUDA if available, else use CPU
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Data transformation for training and validation
    data_transform = {
        "train": transforms.Compose([transforms.Resize(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

        "val": transforms.Compose([transforms.Resize(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # Set up logging, directories, and file paths for confusion matrix and error records
    dir_path = setup_logging_and_dirs(args)

    # Divide the groups based on the patient's individual number.
    groups = df['Path'].apply(lambda x: x.split('/')[6])
    # Use cross-validation
    kfold  = GroupKFold(n_splits=5)

    # Record evaluation metrics for each fold
    accuracies = []
    MAEs = []
    MSEs = []

    for fold, (train_indices, test_indices) in enumerate(kfold.split(images_path, images_label, groups)):

        model = DMIPAN(args).to(device)

        # Split data into training and test sets for this fold
        X_train, X_test = list(images_path[train_indices]), list(images_path[test_indices])
        y_train, y_test = list(images_label[train_indices]), list(images_label[test_indices])

        # Instantiate the training dataset
        train_dataset = MyDataSet(data_list=X_train,
                                  label_list=y_train,
                                  transform=data_transform["train"])

        # Instantiate the validation dataset
        val_dataset = MyDataSet(data_list=X_test,
                                label_list=y_test,
                                transform=data_transform["val"])

        batch_size = args.batch_size
        nw = args.nw
        # Set number of workers for data loading
        print('Using {} dataloader workers every process'.format(nw))

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=nw,
                                                   collate_fn=train_dataset.collate_fn,
                                                   drop_last=True)

        test_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=nw,
                                                 collate_fn=val_dataset.collate_fn,
                                                 drop_last=True)

        # Set optimizer
        pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)

        # Initialize best accuracy and corresponding metrics
        best_accuracy = 0.0
        best_MAE = 5.0
        best_MSE = 0.0

        for epoch in range(args.epochs):
            # Train the model
            train_loss, train_accuracy, train_MAE, train_MSE = train_one_epoch(model=model,
                                                    optimizer=optimizer,
                                                    data_loader=train_loader,
                                                    device=device,
                                                    epoch=epoch)

            # Log training information
            logging.info(
                f"Epoch {epoch}: train_loss = {train_loss:.3f}；train_accuracy = {train_accuracy:.3f}；train_MSE = {train_MSE:.3f}；train_MAE = {train_MAE:.3f}")

            # Evaluate the model on the test set
            test_loss, test_accuracy, test_MAE, test_MSE = evaluate(model=model,
                                                    data_loader=test_loader,
                                                    device=device,
                                                    epoch=epoch)

            # Log evaluation information
            logging.info(
                f"Epoch {epoch}: test_loss = {test_loss:.3f}；test_accuracy = {test_accuracy:.3f}；test_MSE = {train_MSE:.3f}；test_MAE = {test_MAE:.3f}")

            # Check if this is the best accuracy so far
            if test_MAE < best_MAE:
                best_accuracy = test_accuracy
                best_MSE = test_MSE
                best_MAE = test_MAE

        print(f"Fold {fold + 1} test metrics, best_accuracy = {best_accuracy:.3f}；best_MSE = {best_MSE:.3f}；best_MAE = {best_MAE:.3f}")
        # Log the validation results for this fold
        logging.info(f"Fold {fold + 1} test metrics, best_accuracy = {best_accuracy:.3f}；best_MSE = {best_MSE:.3f}； best_MAE = {best_MAE:.3f}")

        accuracies.append(round(best_accuracy, 4))
        MSEs.append(round(best_MSE, 4))
        MAEs.append(round(best_MAE, 4))

    # Calculate the average and standard deviation for each metric across all folds
    average_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    average_MSE = np.mean(MSEs)
    std_MSE = np.std(MSEs)

    average_MAE = np.mean(MAEs)
    std_MAE = np.std(MAEs)

    # Print the results
    print(accuracies)
    print(MSEs)
    print(MAEs)

    print(f"Average accuracy: {average_accuracy:.2f}±{std_accuracy:.2f}; Average MSE: {average_MSE:.2f}±{std_MSE:.2f}; Average MAE: {average_MAE:.2f}±{std_MAE:.2f}")
    # Log the results
    logging.info(f"Accuracies across folds: {accuracies}")
    logging.info(f"MSE across folds: {MSEs}")
    logging.info(f"MAE across folds: {MAEs}")
    logging.info(f"Average accuracy: {average_accuracy:.2f}±{std_accuracy:.2f}; Average MSE: {average_MSE:.2f}±{std_MSE:.2f}; Average MAE: {average_MAE:.2f}±{std_MAE:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--nw', type=float, default=16)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--record', default=True, help='whether to generate log and tensorboard records')

    # optimizer settings
    parser.add_argument('-o', '--optimizer',
                        default="AdamW", type=str, metavar='Opti')
    parser.add_argument('--lr', '--learning_rate',
                        default=0.0001, type=float, metavar='LR', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
    parser.add_argument('--wd', '--weight_decay', default=0.05,
                        type=float, metavar='W', dest='weight_decay')
    parser.add_argument('--eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')

    #
    parser.add_argument('--num_frames', default=80, type=int, help='Number of frames')
    parser.add_argument('--instance_length', default=8,
                        type=int, metavar='N', help='instance length')
    parser.add_argument('--num_classes', default=1, type=int)

    opt = parser.parse_args()

    main(opt)
