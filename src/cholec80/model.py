import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
from sgconv_standalone import GConv
from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP


class SGCONV(pl.LightningModule):
    def __init__(self, config ={},  n_class=7):
        super().__init__()

        self.n_class = n_class
        self.hidden_size=config['hidden_size']
        self.gconv = GConv(
                d_model=self.hidden_size,
                d_state=self.hidden_size,
                l_max=6000,
                bidirectional=True,
                kernel_dim=64,
                n_scales=None,
                decay_min=2,
                decay_max=2
                )

        self.class_head = nn.Sequential(nn.Linear(config['hidden_size'], config['fc1_size'], nn.LeakyReLU()),
                                        nn.Linear(config['fc1_size'], config['fc2_size'], nn.LeakyReLU()),
                                        nn.Linear(config['fc2_size'], 7))

    def forward(self, x = None, y = None):
        # in lightning, forward defines the prediction/inference actions
        batch_size = x.shape[0]
        out_y, out_k = self.gconv(x)
        y_out = []

        for t in range(x.shape[-1]):
            y_hat = self.class_head(out_y[:,:,t])
            y_out.append(y_hat.view(batch_size, 1, -1))
        
        y_out = torch.cat(y_out, dim=1).transpose(1,2)
        return y_out


class LSTM(pl.LightningModule):
    def __init__(self, config ={},  n_class=7):
        super().__init__()

        self.n_class = n_class
        self.hidden_size=config['hidden_size']
        self.gconv = GConv(
                d_model=self.hidden_size,
                d_state=self.hidden_size,
                l_max=10000,
                bidirectional=True,
                kernel_dim=32,
                n_scales=None,
                decay_min=2,
                decay_max=2,
                )

        self.class_head = nn.Sequential(nn.Linear(config['hidden_size'], config['fc1_size'], nn.LeakyReLU()),
                                        nn.Linear(config['fc1_size'], config['fc2_size'], nn.LeakyReLU()),
                                        nn.Linear(config['fc2_size'], 7))

    def forward(self, x = None, y = None):
        # in lightning, forward defines the prediction/inference actions
        batch_size = x.shape[0]
        out_y, out_k = self.gconv(x)
        y_out = []

        for t in range(x.shape[-1]):
            y_hat = self.class_head(out_y[:,:,t])
            y_out.append(y_hat.view(batch_size, 1, -1))
        
        y_out = torch.cat(y_out, dim=1).transpose(1,2)
        return y_out


class FC(pl.LightningModule):
    def __init__(self, config ={},  n_class=7):
        super().__init__()

        self.n_class = n_class
        self.hidden_size=config['hidden_size']
        self.class_head = nn.Sequential(nn.Linear(config['hidden_size'], config['fc1_size'], nn.LeakyReLU()),
                                        nn.Linear(config['fc1_size'], config['fc2_size'], nn.LeakyReLU()),
                                        nn.Linear(config['fc2_size'], config['fc2_size'], nn.LeakyReLU()),
                                        nn.Linear(config['fc2_size'], config['fc2_size'], nn.LeakyReLU()),
                                        nn.Linear(config['fc2_size'], self.n_class))

    def forward(self, x = None, y = None):
        # in lightning, forward defines the prediction/inference actions
        batch_size = x.shape[0]
        y_out = []
        for t in range(x.shape[-1]):
            y_hat = self.class_head(x[:,:,t])
            y_out.append(y_hat.view(batch_size, 1, -1))
        
        y_out = torch.cat(y_out, dim=1).transpose(1,2)
        return y_out


class CFC_Model(pl.LightningModule):
    def __init__(self, config ={},  n_class=7):
        super().__init__()

        self.n_class = n_class
        self.hidden_size=config['hidden_size']
        if config.get('wiring', 'None'):
            wiring = AutoNCP(200, 32)
            in_features = 2048
            self.temporal_model = CfC(in_features, wiring, batch_first=True)
            self.class_head = nn.Sequential(nn.Linear(32, config['fc1_size'], nn.LeakyReLU()),
                                            nn.Linear(config['fc1_size'], config['fc2_size'], nn.LeakyReLU()),
                                            nn.Linear(config['fc2_size'], 7))
        else:
            wiring = None
            self.temporal_model = CfC(config['input_size'],
                config['hidden_size'],
                mode="no_gate" if "gate" in config['argument'] else "default",
                return_sequences=True,
                batch_first=True,
                mixed_memory="mmrnn" in config['argument'],
                backbone_layers=1,
                backbone_units=256,
                backbone_dropout=0.5,
                )

            self.class_head = nn.Sequential(nn.Linear(config['hidden_size'], config['fc1_size'], nn.LeakyReLU()),
                                            nn.Linear(config['fc1_size'], config['fc2_size'], nn.LeakyReLU()),
                                            nn.Linear(config['fc2_size'], 7))


        



    def forward(self, x = None, y = None):
        # in lightning, forward defines the prediction/inference actions
        batch_size = x.shape[0]
        out_y, out_k = self.temporal_model(x.permute(0, 2, 1))
        out_y = out_y.permute(0, 2, 1)
        y_out = []

        for t in range(x.shape[-1]):
            y_hat = self.class_head(out_y[:,:,t])
            y_out.append(y_hat.view(batch_size, 1, -1))
        
        y_out = torch.cat(y_out, dim=1).transpose(1,2)
        return y_out


class NCP_Model(pl.LightningModule):
    def __init__(self, config ={},  n_class=7):
        super().__init__()

        self.n_class = n_class
        self.hidden_size=config['hidden_size']
        wiring = AutoNCP(16, 8)  # 16 units, 1 motor neuron

        self.temporal_model = LTC(config['input_size'], wiring, batch_first=True)
        # self.temporal_model = LTC(config['input_size'], units=512, batch_first=True)
        self.class_head = nn.Sequential(nn.Linear(16, config['fc1_size'], nn.LeakyReLU()),
                                        nn.Linear(config['fc1_size'], config['fc2_size'], nn.LeakyReLU()),
                                        nn.Linear(config['fc2_size'], 7))

    def forward(self, x = None, y = None):
        # in lightning, forward defines the prediction/inference actions
        batch_size = x.shape[0]
        out_y, out_k = self.temporal_model(x.permute(0, 2, 1))
        out_y = out_y.permute(0, 2, 1)
        y_out = []

        for t in range(x.shape[-1]):
            y_hat = self.class_head(out_y[:,:,t])
            y_out.append(y_hat.view(batch_size, 1, -1))
        
        y_out = torch.cat(y_out, dim=1).transpose(1,2)
        return y_out
