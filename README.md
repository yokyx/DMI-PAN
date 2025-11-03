# DMI-PAN
Source code for the pape"From the Lab to the ICU: Video-based Facial Pain  Assessment using a Differential Multiple Instance  Learning Network"

# Abstract:
Pain is a stressor for intensive care unit (ICU) patients, and inadequate pain assessment has been linked to increased morbidity and mortality. Machine learning has demonstrated potential in assisting pain assessment. However, most studies on automated facial pain assessment studies have primarily been limited to laboratory subjects, with only a few targeting hospitalized infants. They are less practical because ICU patients are primarily older adults, and individual variations in illness and pain threshold can lead to long-term and short-term clinical pain. Furthermore, the texture of aging skin and reduced facial muscle elasticity make it challenging to capture pain expression features. To address these challenges, we present an efficient, weakly supervised approach that models video-based facial pain assessment as a multiple-instance learning problem. Specifically, we introduce a new large-scale Intensive Care Unit Pain Expression Dataset (ICUPED), which contains 714 facial videos from 152 ICU patients (mean age: 62.9 years). We propose a novel differential multiple instance pain assessment network (DMIPAN) with a differential learning module (DLM). The network preserves the raw pain information from video snippets via uniform sampling and fine-grained instance partitioning. The DLM uses a pretrained coarse classifier to distinguish long-term from short-term pain expressions. It also models facial pain motion to capture dynamic pain information, thereby reducing individual facial variations. In a comparison with other techniques, our proposed method achieved state-of-the-art (SOTA) results on both the ICUPED dataset and the publicly available UNBC-McMaster Shoulder Pain Archive dataset, improving the accuracy of the ICUPED low-level and high-level pain classification tasks by 11.1% and 5.2%, respectively. The results demonstrate promising potential of our developed DMI-PAN model for automating the process of pain assessment in ICU settings.

# Install and compile the prerequisites：
Python 3.9

PyTorch >= 1.8

NVIDIA GPU + CUDA

Python packages: numpy,pandas,scipy,sklearn

# Pretrained model
You can download the Pseudo-label Generation Classifier pretrained model from the link below.
[Pseudo-label Generation Classifier](https://github.com/yokyx/DMI-PAN/releases/download/v1.0/Classifier_pre_train.pth)

# DATASET
You can access the UNBC-MCMASTER SHOULDER PAIN ARCHIVE dataset by visiting https://www.jeffcohn.net/Resources/ to apply for access. We have uploaded the organized UNBC paths and labels in the UNBC_VAS.xlsx file.

Our dataset ICUPED will be conditionally made available after the article is published. Stay tuned.

