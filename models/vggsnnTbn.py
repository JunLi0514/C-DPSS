from typing import Union, List, Dict, Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
# from spikingjelly.activation_based import functional, layer, surrogate, neuron
from spikingjelly.activation_based import functional, layer, surrogate, neuron



__all__ = [
    "VGG",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "vgg19",
]



class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 10, init_weights: bool = True, dropout: float = 0.5, total_timestep: int = 6,
        affine=True, cfg=None,**kwargs: Any
    ) -> None:
        super().__init__()
        self._AFFINE = affine

        if type(cfg)==str:
            cfg = cfgs[cfg]
        else:
            raise Exception("cfg should be a list but is {}".format(str(cfg)))
 

        # get all index
        inds=[]

        i=0
        for _ , x in enumerate(cfg): 
            if type(x)==int:
                
                if _!= (len(cfg)-1):
                    cfg[_]=x*1
                inds.append(i)
                i+=3
                
            else:
                i+=1

        print (cfg)
        # get split index        
        next_layers = {}
        prev_idx=0

        next_layers_LIF = {}

        for ind in inds: 
            next_layers[prev_idx]=[ind,prev_idx+1]
            next_layers_LIF[prev_idx] = prev_idx+2
            prev_idx=ind

        self.next_layers=next_layers
        self.layer2split = list(next_layers.keys())
        self.next_layers_LIF=next_layers_LIF
            
        print (next_layers,self.layer2split)

        self.total_timestep = total_timestep
        self.features = features
        # self.classifier = nn.Linear(512, num_classes)
        self.classifier = layer.Linear(512, num_classes)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if len(x.shape) == 4:
            x = x.repeat(self.total_timestep, 1,1,1,1)

        x = self.features(x)
        T,B,C,H,W = x.shape
        x = x.reshape(T,B,-1)
        output_list = self.classifier(x)
        return output_list#acc_voltage

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False, _AFFINE=True, store_v_seq=False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            # layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            # layers += [layer.MaxPool2d(kernel_size=2, stride=2, step_mode='m')]
            layers += [Maxpool2d1(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            # conv2d = layer.Conv2d(in_channels, v, kernel_size=3, padding=1, step_mode='m')
            conv2d = Conv2d1(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, 
                           myBatchNorm3d(v),
                           # layer.ThresholdDependentBatchNorm2d(alpha=0.5, v_th=1.0, affine=_AFFINE, num_features=v),
                           # nn.BatchNorm2d(v, affine=_AFFINE),
                           # neuron.ParametricLIFNode(v_threshold=1.0, v_reset=0.0, init_tau=2.,
                           #                          surrogate_function=surrogate.ATan(),
                           #                          detach_reset=True)
                           neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau= 4./3.,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True,
                           step_mode='m', 
                           store_v_seq=store_v_seq)
                           ]
            else:
                layers += [conv2d,
                           # neuron.ParametricLIFNode(v_threshold=1.0, v_reset=0.0, init_tau=2.,
                           #                          surrogate_function=surrogate.ATan(),
                           #                          detach_reset=True)
                           neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau= 4./3.,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True)
                           ]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs["init_weights"] = False
    if 'store_v_seq' in kwargs:
        store_v_seq = kwargs['store_v_seq']
    else:
        store_v_seq = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, store_v_seq=store_v_seq), cfg=cfg, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg11(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg11", "A", False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg11_bn", "A", True, pretrained, progress, **kwargs)


def vgg13(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg13", "B", False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg13_bn", "B", True, pretrained, progress, **kwargs)


def vgg16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg16", "D", False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg16_bn", "D", True, pretrained, progress, **kwargs)


def vgg19(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg19", "E", False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg19_bn", "E", True, pretrained, progress, **kwargs)

class myBatchNorm3d(nn.BatchNorm3d):
    def __init__(self, num_features, eps = 0.00001, momentum = 0.1, affine = True, track_running_stats = True, device=None, dtype=None, step=2):
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)
        # self.bn = nn.BatchNorm3d(num_features)
        self.step = step
    def forward(self, x):
        input = x.permute(1, 2, 0, 3, 4)
        # out = self.bn(out)
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        out = F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
        out = out.permute(2, 0, 1, 3, 4).contiguous()
        return out


class Conv2d1(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, padding_mode = "zeros", device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
    
    def forward(self, x):
        T, B, C, H, W = x.shape
        out = x.reshape(-1, C, H, W)
        out = F.conv2d(out, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        B_o, C_o, H_o, W_o = out.shape
        out = out.view(T, B, C_o, H_o, W_o).contiguous()
        return out

class Maxpool2d1(nn.MaxPool2d):
    def __init__(self, kernel_size, stride = None, padding = 0, dilation = 1, return_indices = False, ceil_mode = False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

    def forward(self, x):
        T, B, C, H, W = x.shape
        out = x.reshape(-1, C, H, W)
        out = F.max_pool2d(out, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, self.return_indices)
        B_o, C_o, H_o, W_o = out.shape
        out = out.view(T, B, C_o, H_o, W_o).contiguous()
        return out

