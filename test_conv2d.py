import mindspore
import torch
import numpy as np
from mindspore import context
context.set_context(mode=context.PYNATIVE_MODE)

conv_ms = mindspore.nn.Conv2d(1024, 234, 1)
conv_pt = torch.nn.Conv2d(1024, 234, 1, bias=False)
count = 0
for i in range(1000):
    weight = np.random.randn(234, 1024, 1, 1).astype(np.float32)
    inputs = np.random.randn(2, 1024, 128, 128).astype(np.float32)
    conv_ms.weight.set_data(mindspore.Tensor(weight))
    conv_pt.weight = torch.nn.parameter.Parameter(torch.tensor(weight), True)
    out_ms = conv_ms(mindspore.Tensor(inputs))
    out_pt = conv_pt(torch.tensor(inputs))

    if np.allclose(out_ms.asnumpy(), out_pt.detach().numpy(), 1e-3, 1e-3):
        count += 1
    print(count)
