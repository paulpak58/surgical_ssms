import argparse
import os
import sys
import pickle
#import tensorflow as tf

from pytorch_lightning.plugins import DDPPlugin

from ncps.torch import CfC
# from cholec80_dataset import Cholec80Dataset

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

from custom_progress import CustomProgress

torch.manual_seed(0)
np.random.seed(0)


class CnnBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv4 = nn.Conv2d(256, 256, 3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        return x


def get_backbone(bn_name):
    backbone = None
    backbone_dim = None
    if bn_name in ["resnet", "resnet_fixed"]:
        # resnet = torchvision.models.resnet18(pretrained=True)
        resnet = torchvision.models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        backbone = nn.Sequential(*modules)
        backbone_dim = 2048  # resnet50
        # backbone_dim = 512  # resnet18
        if bn_name == "resnet_fixed":
            for param in backbone.parameters():
                param.requires_grad = False
    elif bn_name == "cnn":
        backbone = CnnBackbone()
        backbone_dim = 256
    else:
        raise ValueError(f"Unknown backbone '{backbone}'")
    return backbone, backbone_dim


class CNN_Only(nn.Module):
    def __init__(self, backbone, backbone_dim, readout_size, fc1_size=1024, fc_size=512):
        super().__init__()
        self.backbone = backbone
        # self.readout = nn.Linear(backbone_dim, readout_size)

        self.class_head = nn.Sequential(
            nn.Linear(backbone_dim, fc1_size), nn.LeakyReLU(),
            nn.Linear(fc1_size, fc_size), nn.LeakyReLU(),
            nn.Linear(fc_size, readout_size)
        )
        # self.readout.weight.data.fill_(0.0)
        # self.readout.bias.data.fill_(0.0)

        self.class_head.apply(self.__init__weights)

    def __init__weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)


    def forward(self, x):
        size = x.size()
        #print(size,x.shape)
        #x = x.view(size[0] * size[1], size[2], size[3], size[4])
        z = self.backbone(x)
        #z = z.view(size[0], size[1], -1)
        z = z.view(size[0], -1)
        # out = self.readout(z)
        out = self.class_head(z)
        return out


class CNN_RNN(nn.Module):
    def __init__(self, backbone, rnn, rnn_size, readout_size):
        super().__init__()
        self.backbone = backbone
        self.rnn = rnn
        self.readout = nn.Linear(rnn_size, readout_size)

    def forward(self, x):
        size = x.size()
        x = x.view(size[0] * size[1], size[2], size[3], size[4])
        z = self.backbone(x)
        z = z.view(size[0], size[1], -1)
        out, hidden = self.rnn(z)
        out = self.readout(out)
        return out


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
        y_hat = self.model(x)

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
    parser.add_argument("--model", default="lstm")
    parser.add_argument("--backbone", default="resnet")
    parser.add_argument("--opt", default="adamw")
    parser.add_argument("--mmrnn", action="store_true")
    parser.add_argument("--wd", default=1e-6, type=float)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--dr", default=0.1, type=float)
    parser.add_argument("--aug_strength", default=0.3, type=float)
    parser.add_argument("--crop_mode", default=0, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--seq_len", default=60, type=int)  # 1 minute
    parser.add_argument("--size", default=256, type=int)  # 1 minute
    parser.add_argument("--batch_size", default=8, type=int)
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
    # print(f'Length of test_dataset: {len(test_dataset)}')

    # train = Cholec80Dataset(
    #     dataset_dir,
    #     "train",
    #     args.seq_len,
    #     augment=True,
    #     aug_strength=args.aug_strength,
    #     crop_mode=args.crop_mode,
    #     # size=(224, 224),
    #     size=(int(256 * 9 / 16), 256),
    # )
    # val = Cholec80Dataset(
    #     dataset_dir,
    #     "test",
    #     args.seq_len,
    #     augment=False,
    #     # size=(224, 224),
    #     size=(int(256 * 9 / 16), 256),
    # )
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


    backbone, backbone_dim = get_backbone(args.backbone)
    if args.model == "lstm":
        rnn = nn.LSTM(
            input_size=backbone_dim,
            hidden_size=args.size,
            batch_first=True,
        )
        cnn_rnn = CNN_RNN(backbone, rnn, args.size, 7)
    elif args.model.startswith("cfc"):
        rnn = CfC(
            backbone_dim,
            args.size,
            mode="no_gate" if "gate" in args.model else "default",
            return_sequences=False,
            batch_first=True,
            mixed_memory=args.mmrnn,
            backbone_layers=1,
            backbone_units=256,
            backbone_dropout=args.dr,
        )
        cnn_rnn = CNN_RNN(backbone, rnn, args.size, 7)
    elif args.model == "cnn":
        cnn_rnn = CNN_Only(backbone, backbone_dim, 7)
    else:
        raise ValueError("Unknown model")
    model = TemporalTrainer(
        cnn_rnn, class_names=train_dataset.class_names, opt=args.opt, lr=args.lr, wd=args.wd
    )
    progress_bar = CustomProgress(refresh_rate=1)
    trainer = pl.Trainer(
        gpus=-1,
        max_epochs=args.epochs,
        check_val_every_n_epoch=1,
        # precision="bf16",
        # strategy=DDPPlugin(find_unused_parameters=False),
        strategy=pl.strategies.DDPStrategy(find_unused_parameters=False),
        benchmark=True,
        # callbacks=[progress_bar],
        logger=False,
        # gradient_clip_val=1.0,
    )
    # trainer = pl.Trainer(
    #     gpus=-1,
    #     # precision=16,
    #     check_val_every_n_epoch=1,
    #     max_epochs=args.epochs,
    #     strategy=DDPPlugin(find_unused_parameters=False),
    #     progress_bar_refresh_rate=0,
    #     callbacks=[progress_bar],
    #     logger=False,
    #     # default_root_dir="",
    # )
    # trainer.validate(model,dataloader_test)

    trainer.fit(model, dataloader_train, dataloader_val)

    # print("Testing")
    # trainer.test(model, dataloader_test)

    if trainer.is_global_zero:
        cmd = " ".join(sys.argv)
        print(f"Jobs was:\n{cmd}")
        basepath = ""
        if os.path.isdir("/local/mlech_cholec80/"):
            basepath = "/data/"
        with open(basepath + "global_cholec80_results.txt", "a") as f:
            f.write(f"{cmd} # {progress_bar.best_val_acc:0.3f}\n")
# hello ssh