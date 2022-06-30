import torch
from models.segformer import WeTr

wetr = WeTr('mit_b1', num_classes=20, embedding_dim=256, pretrained=False)
#wetr.get_param_groupsv()
dummy_input = torch.rand(2, 2,3,512,512)
wetr(dummy_input)

