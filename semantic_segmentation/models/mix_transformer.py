import math
import mindspore
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import TruncatedNormal, Normal, Zero, initializer
from .modules import DropPath, ModuleParallel, LayerNormParallel, TokenExchange, Identity, num_parallel
from .utils import to_2tuple


class MLP(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ModuleParallel(nn.Dense(in_features, hidden_features))
        self.dwconv = DWConv(hidden_features)
        self.act = ModuleParallel(act_layer(False))
        self.fc2 = ModuleParallel(nn.Dense(hidden_features, out_features))
        self.drop = ModuleParallel(nn.Dropout(1-drop))

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.cells():
            if isinstance(m, nn.Dense):
                m.weight.set_data(initializer(TruncatedNormal(0.02), m.weight.shape))
                if isinstance(m, nn.Dense) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    m.bias.set_data(Zero(), m.bias.shape)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.group
                m.weight.set_data(initializer(Normal(math.sqrt(2.0 / fan_out)), m.weight.shape))

    def construct(self, x, H, W):
        x = self.fc1(x)
        x = [self.dwconv(x[0], H, W), self.dwconv(x[1], H, W)]
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Cell):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = ModuleParallel(nn.Dense(dim, dim, has_bias=qkv_bias))
        self.kv = ModuleParallel(nn.Dense(dim, dim * 2, has_bias=qkv_bias))
        self.attn_drop = ModuleParallel(nn.Dropout(1-attn_drop))
        self.proj = ModuleParallel(nn.Dense(dim, dim))
        self.proj_drop = ModuleParallel(nn.Dropout(1-proj_drop))

        self.sr_ratio = sr_ratio

        if sr_ratio > 1:
            self.sr = ModuleParallel(nn.Conv2d(dim, dim, sr_ratio, sr_ratio, has_bias=True))
            self.norm = LayerNormParallel(dim)
        self.exchange = TokenExchange()

        self.softmax = ops.Softmax()
        self.bmm = ops.BatchMatMul()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.cells():
            if isinstance(m, nn.Dense):
                m.weight.set_data(initializer(TruncatedNormal(0.02), m.weight.shape))
                if isinstance(m, nn.Dense) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    m.bias.set_data(Zero(), m.bias.shape)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.group
                m.weight.set_data(initializer(Normal(math.sqrt(2.0 / fan_out)), m.weight.shape))

    def construct(self, x, H, W, mask):
        B, N, C = x[0].shape
        q = self.q(x)
        q_list = ()
        for q_ in q:
            q_list += (q_.reshape(B, N, self.num_heads, C // self.num_heads).transpose(0, 2, 1, 3),)

        if self.sr_ratio > 1:
            x_list = ()
            for x_ in x:
                x_list += (x_.transpose(0, 2, 1).reshape(B, C, H, W),)
            x = self.sr(x_list)
            x_list = ()
            for x_ in x:
                x_list += (x_.reshape(B, C, -1).transpose(0, 2, 1),)
            x = self.norm(x_list)
            kv = self.kv(x)
            kv_list = ()
            for kv_ in kv:
                kv_list += (kv_.reshape(B, -1, 2, self.num_heads, C // self.num_heads).transpose(2, 0, 3, 1, 4),)
        else:
            kv = self.kv(x)
            kv_list = ()
            for kv_ in kv:
                kv_list += (kv_.reshape(B, -1, 2, self.num_heads, C // self.num_heads).transpose(2, 0, 3, 1, 4),)

        # print(len(kv_list))
        k_list, v_list = (kv_list[0][0], kv_list[1][0]), (kv_list[0][1], kv_list[1][1])
        attn_list = ()
        for (q_, k_) in zip(q_list, k_list):
            attn_list += (self.bmm(q_, k_.swapaxes(-2, -1)) * self.scale,)
        attn_list_2 = ()
        for attn_ in attn_list:
            attn_list_2 += (self.softmax(attn_),)
        attn = self.attn_drop(attn_list_2)

        x_list = ()
        for (attn_, v_) in zip(attn, v_list):
            x_list += (self.bmm(attn_, v_).swapaxes(1, 2).reshape(B, N, C),)
        x = self.proj(x_list)
        x = self.proj_drop(x)

        if mask is not None:
            x_list = ()
            for (x_, mask_) in zip(x, mask):
                x_list += (x_ * mask_.expand_dims(2),)
            x = self.exchange(x_list, mask, mask_threshold=0.02)

        return x

class PredictorLG(nn.Cell):
    """ Image to Patch Embedding from DydamicVit
    """
    def __init__(self, embed_dim=384):
        super().__init__()
        self.score_nets = nn.CellList([nn.SequentialCell(
            nn.LayerNorm([embed_dim]),
            nn.Dense(embed_dim, embed_dim),
            nn.GELU(False),
            nn.Dense(embed_dim, embed_dim // 2),
            nn.GELU(False),
            nn.Dense(embed_dim // 2, embed_dim // 4),
            nn.GELU(False),
            nn.Dense(embed_dim // 4, 2),
            nn.LogSoftmax(axis=-1)
        ) for _ in range(num_parallel)])

    def construct(self, x):
        outs = ()
        for i in range(num_parallel):
            outs += (self.score_nets[i](x[i]),)
        return outs

class Block(nn.Cell):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=LayerNormParallel, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # self.score = PredictorLG(dim)

        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = ModuleParallel(DropPath(drop_path)) if drop_path > 0. else ModuleParallel(Identity())
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.exchange = TokenExchange()

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.cells():
            if isinstance(m, nn.Dense):
                m.weight.set_data(initializer(TruncatedNormal(0.02), m.weight.shape))
                if isinstance(m, nn.Dense) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    m.bias.set_data(Zero(), m.bias.shape)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.group
                m.weight.set_data(initializer(Normal(math.sqrt(2.0 / fan_out)), m.weight.shape))

    def construct(self, x, H, W, mask=None):
        B = x[0].shape[0]
        # norm1 = self.norm1(x)
        # score = self.score(norm1)
        # mask = [F.gumbel_softmax(score_.reshape(B, -1, 2), hard=True)[:, :, 0] for score_ in score]
        # if mask is not None:
        #     norm = [norm_ * mask_.unsqueeze(2) for (norm_, mask_) in zip(norm, mask)]
        f = self.drop_path(self.attn(self.norm1(x), H, W, mask))
        x_list = ()
        for (x_, f_) in zip (x, f):
            x_list += (x_ + f_,)
        # x = [x_ + f_ for (x_, f_) in zip (x, f)]
        f = self.drop_path(self.mlp(self.norm2(x_list), H, W))
        x_list = ()
        for (x_, f_) in zip (x, f):
            x_list += (x_ + f_,)
        # if mask is not None:
        #     x = self.exchange(x, mask, mask_threshold=0.02)
        return x_list

class OverlapPatchEmbed(nn.Cell):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = ModuleParallel(nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                                             pad_mode='pad', padding=(patch_size[0] // 2, patch_size[0] // 2, patch_size[1] // 2, patch_size[1] // 2)))
        self.norm = LayerNormParallel(embed_dim)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.cells():
            if isinstance(m, nn.Dense):
                m.weight.set_data(initializer(TruncatedNormal(0.02), m.weight.shape))
                if isinstance(m, nn.Dense) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    m.bias.set_data(Zero(), m.bias.shape)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.group
                m.weight.set_data(initializer(Normal(math.sqrt(2.0 / fan_out)), m.weight.shape))

    def construct(self, x):
        x = self.proj(x)
        _, _, H, W = x[0].shape
        x_list = ()
        for x_ in x:
            x_list += (x_.reshape(x_.shape[0], x_.shape[1], -1).swapaxes(1, 2),)
        # x = [x_.reshape(x_.shape[0], x_.shape[1], -1).swapaxes(1, 2) for x_ in x]
        x = self.norm(x_list)
        return x, H, W

class MixVisionTransformer(nn.Cell):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=LayerNormParallel,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        predictor_list = [PredictorLG(embed_dims[i]) for i in range(len(depths))]
        self.score_predictor = nn.CellList(predictor_list)

        # transformer encoder
        dpr = [x for x in np.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.CellList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.CellList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.CellList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.CellList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.softmax = ops.Softmax(2)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.cells():
            if isinstance(m, nn.Dense):
                m.weight.set_data(initializer(TruncatedNormal(0.02), m.weight.shape))
                if isinstance(m, nn.Dense) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    m.bias.set_data(Zero(), m.bias.shape)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.group
                m.weight.set_data(initializer(Normal(math.sqrt(2.0 / fan_out)), m.weight.shape))
    '''
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
    '''
    def reset_drop_path(self, drop_path_rate):
        dpr = [x for x in np.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Dense(self.embed_dims, num_classes) if num_classes > 0 else Identity()

    def forward_features(self, x):
        B = x[0].shape[0]
        outs0, outs1 = (), ()
        masks = ()

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            mask = ()
            score = self.score_predictor[0](x)
            for score_ in score:
                mask += (self.softmax(score_.reshape(B, -1, 2))[:, :, 0],)
            # mask = [self.softmax(score_.reshape(B, -1, 2))[:, :, 0] for score_ in score]  # mask_: [B, N]
            masks += (mask,)
            x = blk(x, H, W, mask)
        x = self.norm1(x)
        outs = ()
        for x_ in x:
            outs += (x_.reshape(B, H, W, -1).transpose(0, 3, 1, 2),)
        outs0 += (outs[0],)
        outs1 += (outs[1],)

        # stage 2
        x, H, W = self.patch_embed2(outs)
        for i, blk in enumerate(self.block2):
            mask = ()
            score = self.score_predictor[1](x)
            for score_ in score:
                mask += (self.softmax(score_.reshape(B, -1, 2))[:, :, 0],)
            masks += (mask,)
            x = blk(x, H, W, mask)
        x = self.norm2(x)
        outs = ()
        for x_ in x:
            outs += (x_.reshape(B, H, W, -1).transpose(0, 3, 1, 2),)
        outs0 += (outs[0],)
        outs1 += (outs[1],)

        # stage 3
        x, H, W = self.patch_embed3(outs)
        for i, blk in enumerate(self.block3):
            mask = ()
            score = self.score_predictor[2](x)
            for score_ in score:
                mask += (self.softmax(score_.reshape(B, -1, 2))[:, :, 0],)
            masks += (mask,)
            x = blk(x, H, W, mask)
        x = self.norm3(x)
        outs = ()
        for x_ in x:
            outs += (x_.reshape(B, H, W, -1).transpose(0, 3, 1, 2),)
        outs0 += (outs[0],)
        outs1 += (outs[1],)

        # stage 4
        x, H, W = self.patch_embed4(outs)
        for i, blk in enumerate(self.block4):
            mask = ()
            score = self.score_predictor[3](x)
            for score_ in score:
                mask += (self.softmax(score_.reshape(B, -1, 2))[:, :, 0],)
            masks += (mask,)
            x = blk(x, H, W, mask)
        x = self.norm4(x)
        outs = ()
        for x_ in x:
            outs += (x_.reshape(B, H, W, -1).transpose(0, 3, 1, 2),)
        outs0 += (outs[0],)
        outs1 += (outs[1],)

        return (outs0, outs1), masks

    def construct(self, x):
        x, masks = self.forward_features(x)
        return x, masks


class DWConv(nn.Cell):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 'pad', 1, has_bias=True, group=dim)

    def construct(self, x, H, W):
        B, N, C = x.shape
        x = x.swapaxes(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).swapaxes(1, 2)

        return x

class mit_b0(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=LayerNormParallel, depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b1(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=LayerNormParallel, depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b2(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=LayerNormParallel, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b3(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=LayerNormParallel, depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b4(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=LayerNormParallel, depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b5(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=LayerNormParallel, depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)