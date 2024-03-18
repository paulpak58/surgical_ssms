import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from torchvision.transforms import Lambda
from sklearn import metrics

from transforms import Cholec80Dataset
from transforms import RandomCrop, RandomHorizontalFlip, ColorJitter, RandomRotation 



#########################
# Sampler for Dataloader
#########################
class SeqSampler(Sampler):
    def __init__(self, data_source, idx):
        super().__init__(data_source)
        self.data_source = data_source
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)
    

#########################################################
# Build train, validation, and test Cholec80 datasets
#########################################################
def get_train_val_datasets(seq_length, label_path, data_path, use_flip, crop_type):
    with open(label_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)
    # Extract train, val, and test data, labels, and nums 
    train_x = train_test_paths_labels[0]
    val_x = train_test_paths_labels[1]
    train_labels = train_test_paths_labels[2]
    val_labels = train_test_paths_labels[3]
    train_num_each = train_test_paths_labels[4]
    val_num_each = train_test_paths_labels[5]
    test_x = train_test_paths_labels[6]
    test_labels = train_test_paths_labels[7]
    test_num_each = train_test_paths_labels[8]

    # Prepare labels and transforms
    train_labels = np.asarray(train_labels, dtype=np.int64)
    val_labels = np.asarray(val_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)
    train_transforms = None
    test_transforms = None

    # Train Data Augmentations
    if use_flip == 0:
        train_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            RandomCrop(size=224, sequence_length=seq_length),
            RandomHorizontalFlip(sequence_length=seq_length),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
        ])
    elif use_flip == 1:
        train_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            RandomCrop(size=224, sequence_length=seq_length),
            ColorJitter(sequence_length=seq_length, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            RandomHorizontalFlip(sequence_length=seq_length),
            RandomRotation(degrees=5, sequence_length=seq_length),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
        ])

    # Test Data Augmentations
    if crop_type == 0:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
        ])
    elif crop_type == 1:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
        ])
    elif crop_type == 2:
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
        ])
    elif crop_type == 5:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.FiveCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])(crop)
                     for crop in crops]))
        ])
    elif crop_type == 10:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.TenCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])(crop)
                     for crop in crops]))
        ])

    # Given data, label, and transforms, create our datasets
    # This assumes that the train_val_paths_labels1.pkl from the Google Drive is being used
    # Since they use relative paths, we replace with user data paths
    parent_path = '/'.join(data_path.split('/')[:-1])
    for i in range(len(train_x)):
        train_x[i] = train_x[i].replace('../..', parent_path)
    for i in range(len(val_x)):
        val_x[i] = val_x[i].replace('../..', parent_path)
    for i in range(len(test_x)):
        test_x[i] = test_x[i].replace('../..', parent_path)
    train_dataset = Cholec80Dataset(train_x, train_labels, seq_length, train_transforms)
    val_dataset = Cholec80Dataset(val_x, val_labels, seq_length, test_transforms)
    test_dataset = Cholec80Dataset(test_x, test_labels, seq_length, test_transforms)

    # Return both the datasets and the indices
    return (train_dataset, train_num_each,\
            val_dataset, val_num_each,\
            test_dataset, test_num_each)


#########################################################
# Build train, validation, and test dataloaders
#########################################################
def get_dataloader(
    data_path,
    label_path,
    seq_length,
    train_batch_size,
    val_batch_size,
    use_flip,
    crop_type,
    num_workers=3
):

    train_dataset, train_num_each,\
        val_dataset, val_num_each,\
        test_dataset, test_num_each = get_train_val_datasets(seq_length, label_path, data_path, use_flip, crop_type)


    #########################################################
    # Helper to retrieve start indices for each time sequence
    #########################################################
    def get_start_idx(seq_length, batch_size):
        count = 0
        idx = []
        for i in range(len(batch_size)):
            for j in range(count, count + batch_size[i] - seq_length + 1):
                idx.append(j)
            count += batch_size[i]
        return idx

    # Get start indices for each time sequence
    train_start_idx = get_start_idx(seq_length, train_num_each)
    val_start_idx = get_start_idx(seq_length, val_num_each)
    test_start_idx = get_start_idx(seq_length, test_num_each)

    # Get global indices for each time sequence    
    train_idx, val_idx, test_idx = [], [], []
    for i in range(len(train_start_idx)):
        for j in range(seq_length):
            train_idx.append(train_start_idx[i] + j)
    for i in range(len(val_start_idx)):
        for j in range(seq_length):
            val_idx.append(val_start_idx[i] + j)
    for i in range(len(test_start_idx)):
        for j in range(seq_length):
            test_idx.append(test_start_idx[i] + j)

    num_train = len(train_idx)
    num_val = len(val_idx)
    num_test = len(test_idx)
    print('Size of training data: ', num_train)
    print('Size of validation data: ', num_val)
    print('Size of test data: ', num_test)

    # Build our dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=SeqSampler(train_dataset,train_idx),
        num_workers=num_workers,
        pin_memory=False
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        sampler=SeqSampler(val_dataset,val_idx),
        num_workers=num_workers,
        pin_memory=False
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        sampler=SeqSampler(test_dataset,test_idx),
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_dataloader, val_dataloader, test_dataloader

if __name__=='__main__':
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(
        data_path='',
        label_path='/home/ppak/surgical_adventure/src/Trans-SVNet/train_val_paths_labels1.pkl',
        seq_length=1,
        train_batch_size=400,
        val_batch_size=400,
        use_flip=1,
        crop_type=1,
        num_workers=4
    )
    print(f'Length of train_dataloader: {len(train_dataloader)}')
    print(f'Length of val_dataloader: {len(val_dataloader)}')
    print(f'Length of test_dataloader: {len(test_dataloader)}')