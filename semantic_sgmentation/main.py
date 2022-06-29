from models.segformer import WeTr
import mindspore.numpy as mnp
from mindspore import context

context.set_context(mode=context.PYNATIVE_MODE)
wetr = WeTr('mit_b1', num_classes=20, embedding_dim=256, pretrained=True)
dummy_input = mnp.rand(2,2,3,512,512)
wetr(dummy_input)