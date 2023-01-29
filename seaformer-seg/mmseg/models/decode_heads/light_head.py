from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *
import torch.nn as nn
import torch.nn.functional as F


class Fusion_block(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            embed_dim: int,
            norm_cfg=dict(type='BN', requires_grad=True),
            activations=None,
    ) -> None:
        super(Fusion_block, self).__init__()
        self.norm_cfg = norm_cfg
        self.local_embedding = ConvModule(inp, embed_dim, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_act = ConvModule(oup, embed_dim, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.act = h_sigmoid()

    def forward(self, x_l, x_g):
        '''
        x_g: global features
        x_l: local features
        '''
        B, C, H, W = x_l.shape
        B, C_c, H_c, W_c = x_g.shape
      
        local_feat = self.local_embedding(x_l)
        global_act = self.global_act(x_g)
        sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
        out = local_feat * sig_act
        return out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


@HEADS.register_module()
class LightHead(BaseDecodeHead):
    """
    SEA-Former: Squeeze-enhanced Axial Transformer for Mobile Semantic Segmentation
    """
    def __init__(self, embed_dims, is_dw=False, **kwargs):
        super(LightHead, self).__init__(input_transform='multiple_select', **kwargs)

        head_channels = self.channels
        in_channels = self.in_channels    
        self.linear_fuse = ConvModule(
            in_channels=head_channels,
            out_channels=head_channels,
            kernel_size=1,
            stride=1,
            groups=head_channels if is_dw else 1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        for i in range(len(embed_dims)):
            fuse = Fusion_block(in_channels[0] if i == 0 else embed_dims[i-1], in_channels[i+1], embed_dim=embed_dims[i], norm_cfg=self.norm_cfg)
            setattr(self, f"fuse{i + 1}", fuse)
        self.embed_dims = embed_dims
        
    def forward(self, inputs):
        xx = self._transform_inputs(inputs)  
        x_detail = xx[0]
        for i in range(len(self.embed_dims)):
            fuse = getattr(self, f"fuse{i + 1}")
            x_detail = fuse(x_detail, xx[i+1])
        _c = self.linear_fuse(x_detail)
        x = self.cls_seg(_c)
        return x
