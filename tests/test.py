import unittest
import torch
import mindspore
import numpy as np
from mindspore import load_param_into_net
from semantic_segmentation_torch.models.segformer import WeTr as WeTr_pt
from semantic_segmentation.models.segformer import WeTr as WeTr_ms
from convert.torch2ms import torch2ms

class TestWeTr(unittest.TestCase):
    def setUp(self):
        self.dummy_input = np.random.randn(2, 2, 3, 512, 512).astype(np.float32)

    def test_wetr_pytorch(self):
        wetr = WeTr_pt('mit_b1', num_classes=20, embedding_dim=256, pretrained=False)
        dummy_input = torch.tensor(self.dummy_input)
        wetr(dummy_input)

    def test_wetr_mindspore(self):
        wetr = WeTr_ms('mit_b1', num_classes=20, embedding_dim=256, pretrained=False)
        dummy_input = mindspore.Tensor(self.dummy_input)
        wetr(dummy_input)

    def test_wetr_torch2ms(self):
        wetr_pt = WeTr_pt('mit_b1', num_classes=20, embedding_dim=256, pretrained=False)
        wetr_pt_states_num = len(wetr_pt.state_dict())

        wetr_ms = WeTr_ms('mit_b1', num_classes=20, embedding_dim=256, pretrained=False)
        wetr_ms_states_num = len(wetr_ms.parameters_dict())

        ms_states = torch2ms(wetr_pt.state_dict())
        # print(set(ms_states.keys()) - set(wetr_ms.parameters_dict().keys()))
        # print(wetr_pt_states_num, wetr_ms_states_num)

        # ignore num_batches_tracked
        assert wetr_pt_states_num - 1== wetr_ms_states_num
        not_loaded = load_param_into_net(wetr_ms, ms_states)
        print(not_loaded)
        assert not not_loaded

    def test_wetr_torch2ms_precision(self):
        from mindspore import context
        context.set_context(mode=context.PYNATIVE_MODE)
        wetr_pt = WeTr_pt('mit_b1', num_classes=20, embedding_dim=256, pretrained=False)

        wetr_ms = WeTr_ms('mit_b1', num_classes=20, embedding_dim=256, pretrained=False)

        ms_states = torch2ms(wetr_pt.state_dict())

        not_loaded = load_param_into_net(wetr_ms, ms_states)
        assert not not_loaded

        wetr_pt.eval()
        wetr_ms.set_train(False)

        dummy_input_pt = torch.tensor(self.dummy_input)
        dummy_input_ms = mindspore.Tensor(self.dummy_input)

        try:
            outputs_ms, masks_ms = wetr_ms(dummy_input_ms)
        except:
            pass
        try:
            outputs_pt, masks_pt = wetr_pt(dummy_input_pt)
        except:
            pass
        # print(outputs_pt[0].detach().numpy() - outputs_ms[0].asnumpy())
        assert np.allclose(outputs_pt[0].detach().numpy(), outputs_ms[0].asnumpy(), 1e-3, 1e-3)
