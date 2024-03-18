import argparse
import os
import sys
import pickle
#import tensorflow as tf

from pytorch_lightning.plugins import DDPPlugin
#from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import ModelCheckpoint

from ncps.torch import CfC

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR

from types import SimpleNamespace

import numpy as np
import pytorch_lightning as pl
from torchmetrics.functional.classification import multiclass_average_precision
from timm.optim import create_optimizer

# Swin
import transformers
from transformers import SwinModel

torch.manual_seed(0)
np.random.seed(0)



def compute_mAPs(preds, labels):
    aps = multiclass_average_precision(preds, labels, num_classes=7, average=None)
    individual_aps = {f"phase_{i}": aps[i] for i in range(7)}
    mean_ap = torch.mean(aps)
    return mean_ap, individual_aps


class TemporalTrainer(pl.LightningModule):
    def __init__(self, model, class_names=[], opt="adam", lr=0, wd=0):
        super().__init__()
        self.model = model
        self._lr = lr
        self._opt_name = opt
        self._wd = wd
        self.class_names = class_names
        self.criterion = nn.CrossEntropyLoss()

        weights_train = np.asarray(
            [
                1.6411019141231247,
                0.19090963801041133,
                1.0,
                0.2502662616859295,
                1.9176363911137977,
                0.9840248158200853,
                2.174635818337618,
            ]
        )

        self.criterion = nn.CrossEntropyLoss(
            weight=torch.from_numpy(weights_train).float()
        )

        self.n_class = len(self.class_names)

        #conv_params = [p for name, p in self.model.named_parameters() if "fc" not in name]
        #fc_params = [p for name, p in self.model.named_parameters() if "fc" in name]
        # self.optimizer_params = [
        #     {"params": conv_params, "lr": self._lr},        # default: 5e-4
        #     {"params": fc_params, "lr": self._lr / 10},     # default: 5e-5
        # ]

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        y_hat = self.model(x)

        # Last step only
        # if len(y_hat.size()) > 2:
        #     y_hat = y_hat[:, -1]
        # y = y[:, -1]

        loss = self.criterion(y_hat, y)
        # preds = torch.argmax(y_hat.detach(), dim=-1).flatten()
        preds = torch.argmax(y_hat, dim=-1).flatten()
        #labels = torch.argmax(y.detach(), dim=-1).flatten()
        labels = y
        acc = (preds == labels).float().mean()
        self.log("train_acc", acc, prog_bar=True)
        # labels = torch.argmax(y.detach(), dim=-1).flatten()
        # preds = y_hat.detach().view(-1, 7)
        # ap = multiclass_average_precision(preds, labels, num_classes=7)
        # mAP, individual_aps = compute_mAPs(preds, labels)
        # self.log("mAP", mAP, prog_bar=True)
        # for k, v in individual_aps.items():
        #     self.log(k, v, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_validation_start(self) -> None:
        # define cm
        self.cm = torch.zeros(self.n_class, self.n_class)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        print(f'input shape {x.shape}')
        y_hat = self.model(x)
        print(f'output shape {y_hat.shape}')

        # Last step only
        # if len(y_hat.size()) > 2:
        #     y_hat = y_hat[:, -1]
        # y = y[:, -1]

        loss = self.criterion(y_hat, y)
        #preds = torch.argmax(y_hat.detach(), dim=-1).flatten()
        preds = torch.argmax(y_hat, dim=-1).flatten()
        #labels = torch.argmax(y.detach(), dim=-1).flatten()
        #labels = y.detach()
        labels = y
        acc = (preds == labels).float().mean()
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        # labels = torch.argmax(y.detach(), dim=-1).flatten()
        # preds = y_hat.detach().view(-1, 7)
        # ap = multiclass_average_precision(preds, labels, num_classes=7)
        # self.log("val_mAP", ap, prog_bar=True)
        # mAP, individual_aps = compute_mAPs(preds, labels)
        # self.log("val_mAP", mAP, prog_bar=True)
        # not_nan_AP = []
        # for k, v in individual_aps.items():
        #     if torch.isfinite(v).item():
        #         self.log("val_" + k, v, prog_bar=True)
        #         not_nan_AP.append(v)
        # self.log("val_mAP", torch.mean(torch.Tensor(not_nan_AP)), prog_bar=True)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        # for idx_batch in range(y.shape[0]):
        #     gt = y[idx_batch, -1].argmax(dim=-1)
        #     est = y_hat[idx_batch, -1].argmax(dim=-1)
        #     self.cm[int(gt), int(est)] += 1.0
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        #preds = torch.argmax(y_hat.detach(), dim=-1).flatten()
        preds = torch.argmax(y_hat, dim=-1).flatten()
        #labels = y.detach()
        labels = y
        acc = (preds == labels).float().mean()
        self.log("test_acc", acc, prog_bar=True, sync_dist=True)

    # def on_validation_end(self) -> None:
    #     if self.trainer.is_global_zero:
    #         cm = self.cm.detach().cpu().numpy()
    #         accuracy = cm.diagonal() / cm.sum(axis=0)
    #         accuracy[np.isnan(accuracy)] = 0.0
    #         for idx, class_name in enumerate(self.class_names):
    #             print(class_name + " :" + str(accuracy[idx]))
    #         accuracy_mean = cm.diagonal().sum() / cm.flatten().sum()
    #         print("Overall" + " :" + str(accuracy_mean))

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(
        #     filter(lambda p: p.requires_grad, self.parameters()), lr=self._lr
        # )
        args = SimpleNamespace()
        args.weight_decay = self._wd
        args.lr = self._lr

        args.opt = self._opt_name
        args.momentum = 0.9

        optimizer = create_optimizer(args, self)
        # divide lr in half every 5 epochs
        lr_scheduler = StepLR(optimizer, 10, 0.5)
        
        return [optimizer], [lr_scheduler]
        # if args.opt=='adamw':
        #     optimizer = torch.optim.AdamW(self.optimizer_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)  # Add momentum terms
        # else:
        #     optimizer = torch.optim.Adam(self.optimizer_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)  # Add momentum terms
        # return [optimizer]



if __name__ == "__main__":
    ###python train_ncp.py --track_name phase --data_dir DATA_PATH/videos/
    # --annotation_filename DATA_PATH/annotations/ --temporal_length 8 --sampling_rate 1 --cache_dir ./cache --num_dataloader_workers 8 --num_epochs 20

    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="resnet")
    parser.add_argument("--opt", default="adamw")
    parser.add_argument("--wd", default=0.05, type=float)
    parser.add_argument("--lr", default=0.00004, type=float)
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--batch_size", default=75, type=int)
    parser.add_argument("--label_path", default='/home/ppak/surgical_ncp/train_val_paths_labels1.pkl', type=str)
    args = parser.parse_args()

    from transforms import RandomCrop, ColorJitter, RandomHorizontalFlip, RandomRotation, CholecDataset
    with open(args.label_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)
    train_paths_80 = train_test_paths_labels[0]
    val_paths_80 = train_test_paths_labels[1]
    train_labels_80 = train_test_paths_labels[2]
    val_labels_80 = train_test_paths_labels[3]
    train_num_each_80 = train_test_paths_labels[4]
    val_num_each_80 = train_test_paths_labels[5]
    test_paths_80 = train_test_paths_labels[6]
    test_labels_80 = train_test_paths_labels[7]
    test_num_each_80 = train_test_paths_labels[8]
    parent_path = '/'.join(args.label_path.split('/')[:-1])
    for i in range(len(train_paths_80)):
        train_paths_80[i] = train_paths_80[i].replace('../..', parent_path)
    for i in range(len(val_paths_80)):
        val_paths_80[i] = val_paths_80[i].replace('../..', parent_path)
    for i in range(len(test_paths_80)):
        test_paths_80[i] = test_paths_80[i].replace('../..', parent_path)
    train_labels_80 = np.asarray(train_labels_80, dtype=np.int64)
    val_labels_80 = np.asarray(val_labels_80, dtype=np.int64)
    test_labels_80 = np.asarray(test_labels_80, dtype=np.int64)

    # Create train and test transforms
    sequence_length = 1
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((250, 250)),
        RandomCrop(size=224, sequence_length=sequence_length),
        ColorJitter(sequence_length=sequence_length, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        RandomHorizontalFlip(sequence_length=sequence_length),
        RandomRotation(degrees=5, sequence_length=sequence_length),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
    ])
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((250, 250)),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
    ])
    train_dataset = CholecDataset(train_paths_80, train_labels_80, train_transforms)
    val_dataset = CholecDataset(val_paths_80, val_labels_80, test_transforms)
    test_dataset = CholecDataset(test_paths_80, test_labels_80, test_transforms)

    # Combined val, test dataset
    val_dataset = torch.utils.data.ConcatDataset([val_dataset, test_dataset])
    print(f'Length of train_dataset: {len(train_dataset)}')
    print(f'Length of val_dataset: {len(val_dataset)}')

    dataloader_train = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True, 
        shuffle=True,
        num_workers=4,
    )
    dataloader_val = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=False,
        num_workers=4,
    )
    # dataloader_test = DataLoader(
    #     test_dataset,
    #     batch_size=args.batch_size,
    #     drop_last=True,
    #     shuffle=False,
    #     num_workers=4,
    # )



    swin_model = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
    print(swin_model)


    model = TemporalTrainer(
        swin_model, class_names=train_dataset.class_names, opt=args.opt, lr=args.lr, wd=args.wd
    )

    checkpoint_callback = ModelCheckpoint(dirpath="./pl_checkpoints", save_top_k=1, monitor="val_loss")
    #progress_bar = CustomProgress(refresh_rate=1)
    trainer = pl.Trainer(
        gpus=-1,
        max_epochs=args.epochs,
        check_val_every_n_epoch=1,
        # precision="bf16",
        # strategy=DDPPlugin(find_unused_parameters=False),
        strategy=pl.strategies.DDPStrategy(find_unused_parameters=False),
        benchmark=True,
        callbacks=[checkpoint_callback],
        logger=False,
        # gradient_clip_val=1.0,
    )
    trainer.fit(model, dataloader_train, dataloader_val)
    # trainer.test(model, dataloader_test)
    
# hello ssh
