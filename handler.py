import torch
import torch.nn.functional as F
from model import ds_conv, ds_sf, ds_wt, bicubic_pp, bicubic_pp_prunnable
import yaml
import os
import torch.nn as nn


class ModelHandler():

    def __init__(self, config_path):
        self.models = {
            'pretraining': None,
            'prunning': None,
            'bias_removal': None 
        }
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            self.current_mode = 'pretraining' if self.config['pretraining']['enabled'] else \
                                    ('prunning' if self.config['prunning']['enabled'] else \
                                    'bias_removal')
            self.model_save_path = self.config['general']['model_save_path']
            self.model_load_path = self.config['general']['model_load_path']
            self.downsampling = self.config['general']['downsampling']
            self.activation = self.config['general']['activation']
            self.scale = self.config['general']['scale']
            self.m_blocks = self.config['general']['m_blocks']
            self.r_blocks = self.config['general']['r_blocks']
            self.pretraining = self.config['pretraining']['enabled']
            self.prunning = self.config['prunning']['enabled']
            self.bias_removal = self.config['bias_removal']['enabled']
            self.device = self.config['general']['device']
            self.activation = self.config['general']['activation']
            self.seed = self.config['general']['seed']
            self.optimizer = self.config['general']['optimizer']
            self.learning_rate = self.config['general']['learning_rate']
            self.epoch = self.config['general']['num_epochs']
            self.device = self.config['general']['device']
            self.start_channels = self.config['pretraining']['num_channels']
            self.end_channels = self.config['prunning']['num_channels']
            self.image_color_channels = self.config['general']['image_color_channels']
            self.padding_mode = self.config['general']['padding_mode']

        self.model = None
        self.optim = None
        self.criterion = F.mse_loss
        self.stage = 'pretraining'

    def build_models(self):

        ds = self.downsampling

        if self.activation == 'relu': act = nn.ReLU()
        elif self.activation == 'prelu': act = nn.PReLU()
        elif self.activation == 'lrelu': act = nn.LeakyReLU()

        self.model = bicubic_pp(scale=self.scale, R=self.r_blocks, ch=self.end_channels, M=self.m_blocks, 
                                ds=ds, ch_in=self.image_color_channels, relu=act, padding_mode=self.padding_mode)
        
        if self.prunning:
            self.model = bicubic_pp_prunnable(scale=self.scale, ch=self.end_channels, 
                                ch_in=self.image_color_channels, relu=act, padding_mode=self.padding_mode)

        # self.models['pretraining'] = self.model


    def set_stage(self, stage):
        self.stage = stage
    
    def get_stage(self):
        return self.stage


    def preparation(self):
        if self.optimizer == 'adam':
            self.optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'sgd':
            self.optim = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.model.to(self.device)


    def train(self, train_loader):
        # self.model.train()
        # for epoch in range(self.epoch):
        #     for data, target in train_loader:
        #         self.optim.zero_grad()
        #         output = self.model(data)
        #         loss = self.criterion(output, target)
        #         loss.backward()
        #         self.optim.step()
        #     print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        self.models[self.stage] = self.model

    def validation(self, val_loader):
        self.model.eval()
        val_psnr = 0
        # with torch.no_grad():
        #     for data, target in val_loader:
        #         output = self.model(data)
        #         mse = self.criterion(output, target)
        #         val_loss += self.get_psnr(mse)
        # val_psnr /= len(val_loader)
        return val_psnr

    def get_psnr(self, value):
        return 10 * torch.log10(1 / value)

    def save(self):
        torch.save(self.models, 
                   os.path.join(self.model_save_path, 'ckpt.pt'))

    def load(self):
        torch.load(self.models, 
                   os.path.join(self.model_load_path, 'ckpt.pt'))
        
    def remove_bias(self):
        self.model = self.models['prunning'] if self.models['prunning'] is not None \
                            else self.models['pretraining']
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                module.bias = None
        state_dict = self.model.state_dict()
        for key in list(state_dict.keys()):
            if "bias" in key:
                del state_dict[key]

    def set_mask(self, x, y):
        assert isinstance(self.model, bicubic_pp_prunnable)
        self.model.set_mask(x, y)

    def channel_removal(self, mask_id, ch):
        assert isinstance(self.model, bicubic_pp_prunnable)
        self.model.prune_layers(mask_id, ch)

    def remove_mask(self):
        assert isinstance(self.model, bicubic_pp_prunnable)
        self.model.remove_mask()
    
    def rollback_model(self):
        if self.stage == 'prunning':
            self.model = self.models['pretraining']
        elif self.stage == 'bias_removal':
            self.model = self.models['prunning'] if self.models['prunning'] is not None \
                            else self.models['pretraining']

