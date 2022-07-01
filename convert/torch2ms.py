import torch
import re
import mindspore
from mindspore import Parameter, Tensor

def torch2ms(state_dict):
    ms_state_dict = {}
    for k, v in state_dict.items():
        if 'bn' in k:
            k = k.replace('weight', 'gamma')
            k = k.replace('bias', 'beta')
            k = k.replace('running_mean', 'moving_mean')
            k = k.replace('running_var', 'moving_variance')        
        if re.search(r'.ln_[0-9].', k):
            k = k.replace('ln_', 'ln_list.')
            k = k.replace('weight', 'gamma')
            k = k.replace('bias', 'beta')
        if re.search(r'.score_nets\.[0-9]\.0.', k):
            k = k.replace('weight', 'gamma')
            k = k.replace('bias', 'beta')
        new_v = Parameter(Tensor(v.numpy(), dtype=mindspore.float32), name=k, requires_grad=v.requires_grad)
        ms_state_dict[k] = new_v
    
    return ms_state_dict