import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

num_parallel = 2

class TokenExchange(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x, mask, mask_threshold):
        # x0, x1 = ops.zeros_like(x[0]), ops.zeros_like(x[1])
        # x0[mask[0] >= mask_threshold] = x[0][mask[0] >= mask_threshold]
        # x0[mask[0] < mask_threshold] = x[1][mask[0] >= mask_threshold]
        # x1[mask[1] >= mask_threshold] = x[1][mask[1] >= mask_threshold]
        # x1[mask[1] > mask_threshold] = x[0][mask[1] >= mask_threshold]
        mask0 = mask[0].expand_dims(2).expand_as(x[0])
        mask1 = mask[1].expand_dims(2).expand_as(x[1])
        x0 = ops.select(mask0 >= mask_threshold, x[0], x[1])
        x1 = ops.select(mask1 < mask_threshold, x[0], x[1])
        return (x0, x1)

class ModuleParallel(nn.Cell):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def construct(self, x_parallel):
        outs = ()
        for x in x_parallel:
            outs += (self.module(x),)
        return outs

class LayerNormParallel(nn.Cell):
    def __init__(self, num_features):
        super().__init__()
        ln_list = []
        for i in range(num_parallel):
            ln_list.append(nn.LayerNorm([num_features], epsilon=1e-6))
        self.ln_list = nn.CellList(ln_list)

    def construct(self, x_parallel):
        outs = ()
        for i, x in enumerate(x_parallel):
            outs += (self.ln_list[i](x),)
        return outs

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = ops.uniform(shape, Tensor(0.0), Tensor(1.0))
    random_tensor = random_tensor < keep_prob
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor /= keep_prob
    return x * random_tensor

class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def construct(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extend_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class Identity(nn.Cell):    
    def construct(self, x):
        return x

class Dropout2d(nn.Cell):
    def __init__(self, keep_prob=1.0):
        super().__init__()
        self.keep_prob = keep_prob
        self.dropout_2d = ops.Dropout2D(keep_prob)
    
    def construct(self, inputs):
        if not self.training:
            return inputs

        if self.keep_prob == 1:
            return inputs

        out, _ = self.dropout_2d(inputs)
        return out

class ConvModule(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def construct(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out