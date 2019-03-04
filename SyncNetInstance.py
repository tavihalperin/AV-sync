#!/usr/bin/python
#-*- coding: utf-8 -*-
# Video 25 FPS, Audio 16000HZ

import torch
from SyncNetModel import *


cuda = torch.cuda.is_available()

class SyncNetInstance(torch.nn.Module):

    def __init__(self, dropout = 0, num_layers_in_fc_layers = 1024):
        super(SyncNetInstance, self).__init__();
        self.__S__ = S(num_layers_in_fc_layers = num_layers_in_fc_layers)
        if cuda:
            self.__S__ = self.__S__.cuda()

    def loadParameters(self, path):
        loaded_state = torch.load(path, map_location=lambda storage, loc: storage)

        self_state = self.__S__.state_dict();

        for name, param in loaded_state.items():

            self_state[name].copy_(param);
