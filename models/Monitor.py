import math
import os
import time
import torch
import torch.distributed
import errno
import datetime
import random
import numpy as np
from torchvision import transforms
from torch import Tensor, nn
from math import nan
import torch
from models.vggsnnTbn import Conv2d1
from spikingjelly.activation_based import functional, layer, surrogate, neuron

def unpack_len1_tuple(x: tuple or torch.Tensor):
    if isinstance(x, tuple) and x.__len__() == 1:
        return x[0]
    else:
        return x


def snn_to_ann_input(x: torch.Tensor):
    if len(x.shape) == 5:
        return x.flatten(0, 1)
    else:
        return x


class BaseMonitor:
    def __init__(self):
        self.hooks = []
        self.monitored_layers = []
        self.records = []
        self.name_records_index = {}
        self._enable = True

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.records[i]
        elif isinstance(i, str):
            y = []
            for index in self.name_records_index[i]:
                y.append(self.records[index])
            return y
        else:
            raise ValueError(i)

    def clear_recorded_data(self):
        self.records.clear()
        for k, v in self.name_records_index.items():
            v.clear()

    def enable(self):
        self._enable = True

    def disable(self):
        self._enable = False

    def is_enable(self):
        return self._enable

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def __del__(self):
        self.remove_hooks()



class SOPMonitor(BaseMonitor):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.w_nonzero = 0 
        for name, m in net.named_modules():
            if isinstance(m, Conv2d1) or isinstance(m, layer.Linear)\
                or isinstance(m, nn.Conv2d):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                connects = (m.weight!=0).float()
                # connects = torch.ones_like(m.weight,dtype=torch.float32)
                self.w_nonzero += connects.sum()
                self.hooks.append(m.register_forward_hook(self.create_hook(name, connects, mask=None)))

    def cal_sopConv(self, x: Tensor, connects: Tensor, mask: Tensor, m: nn.Conv2d):
        try:
            if x.abs().max()!=1 and x.abs().min()!=0:
                out = torch.nn.functional.conv2d((x!=0).float(), connects, None, m.stride, m.padding, m.dilation,
                                             m.groups)
            else:
                out = torch.nn.functional.conv2d(x, connects, None, m.stride, m.padding, m.dilation,
                                             m.groups)
            # out = torch.nn.functional.conv2d(torch.ones_like(x), connects, None, m.stride, m.padding, m.dilation,
            #                                  m.groups)
        except RuntimeError:
            connects = (m.weight!=0).float()
            if x.abs().max()!=1 and x.abs().min()!=0:
                out = torch.nn.functional.conv2d((x!=0).float(), connects, None, m.stride, m.padding, m.dilation,
                                             m.groups)
            else:
                out = torch.nn.functional.conv2d(x, connects, None, m.stride, m.padding, m.dilation,
                                             m.groups)
 
        if mask is None:
            sop = out.sum()
        else:
            sop = (out * mask).sum()
        return sop.unsqueeze(0)

    def cal_sopLinear(self, x: Tensor, connects: Tensor, mask: Tensor, m: nn.Linear):
        out = torch.nn.functional.linear(x, connects, None)
        # out = torch.nn.functional.linear(torch.ones_like(x), connects, None)
        if mask is None:
            sop = out.sum()
        else:
            sop = (out * mask).sum()
        return sop.unsqueeze(0)


    def create_hook(self, name, connects, mask):
        def hook(m, x: Tensor, y: Tensor):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                if isinstance(m, Conv2d1) :
                    self.records.append(
                        self.cal_sopConv(
                            snn_to_ann_input(unpack_len1_tuple(x)).detach(), connects, mask, m))
                elif isinstance(m, nn.Conv2d):
                    self.records.append(
                        self.cal_sopConv(
                            unpack_len1_tuple(x).detach(), connects, mask, m))
                elif isinstance(m, layer.Linear):
                    self.records.append(
                        self.cal_sopLinear(
                            snn_to_ann_input(unpack_len1_tuple(x)).detach(), connects, mask, m))
 

        return hook

from spikingjelly.activation_based import neuron
class SpikeMonitor(BaseMonitor):
    def __init__(self, net: nn.Module, timestep, num_samples):
        super().__init__()
        self.w_nonzero = 0 
        self.name_records_index_pre = {}
        for name, m in net.named_modules():
            if isinstance(m, neuron.LIFNode):
                self.monitored_layers.append(name)
                self.name_records_index[name] = None 
                self.name_records_index_pre[name] = None 
                self.hooks.append(m.register_forward_hook(self.create_hook(name)))

    def cal_spikes(self, y: Tensor):
        return y.sum(dim=(0,1))


    def create_hook(self, name):
        def hook(m, x: Tensor, y: Tensor):
            if self.is_enable():
                if self.name_records_index[name] == None:
                    self.name_records_index[name] = self.cal_spikes(y)
                else:
                    self.name_records_index[name] += self.cal_spikes(y)

        return hook

    def div_channels(self):
        for name in self.name_records_index:
            _ = self.name_records_index[name].shape
            map_size = _[1] * _[2]
            self.name_records_index[name] = self.name_records_index[name].sum(dim=(1,2))# /map_size

    def reset(self):
        for k,v in self.name_records_index.items():
            self.name_records_index_pre[k] = v
        for name in self.name_records_index:
            if self.name_records_index[name] == None:
                break
            self.name_records_index[name] = None


class MemMonitor(BaseMonitor):
    def __init__(self, net: nn.Module, timestep, num_samples):
        super().__init__()
        self.w_nonzero = 0 
        self.name_records_index_pre = {}
        for name, m in net.named_modules():
            if isinstance(m, neuron.LIFNode):
                self.monitored_layers.append(name)
                self.name_records_index[name] = None 
                self.name_records_index_pre[name] = None 
                self.hooks.append(m.register_forward_hook(self.create_hook(name)))
        self.timestep = timestep
        self.num_samples = num_samples

    def cal_mem(self, y: Tensor):
        return y.sum(dim=(0,1))


    def create_hook(self, name):
        def hook(m, x: Tensor, spike: Tensor):
            if self.is_enable():
                if self.name_records_index[name] == None:
                    self.name_records_index[name] = m.v_seq.abs().sum(dim=(0,1)).detach()/self.timestep*self.num_samples

                else:
                    self.name_records_index[name] += m.v_seq.abs().sum(dim=(0,1)).detach()/self.timestep*self.num_samples

        return hook
    

    def div_channels(self):
        for name in self.name_records_index:
            _ = self.name_records_index[name].shape
            map_size = _[1] * _[2]
            self.name_records_index[name] = self.name_records_index[name].sum(dim=(1,2))# /map_size

    def reset(self):
        for k,v in self.name_records_index.items():
            self.name_records_index_pre[k] = v
        for name in self.name_records_index:
            if self.name_records_index[name] == None:
                break
            self.name_records_index[name] = None
