"""
SUTrack Model
"""
import torch
import math
from torch import nn
import torch.nn.functional as F

from .encoder import build_encoder
from .decoder import build_decoder
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils.pos_embed import get_sinusoid_encoding_table, get_2d_sincos_pos_embed
from.neck import build_neck
from .fem import FEM
from ...config.sutrack.config import cfg


class SUTRACK(nn.Module):
    """ This is the base class for SUTrack """
    def __init__(self, encoder, decoder,neck,fem,
                 num_frames=1, num_template=1,
                 decoder_type="CENTER"):
        """ Initializes the model.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder_type = decoder_type
        self.neck=neck
        self.fem = fem

        self.class_token = False if (encoder.body.cls_token is None) else True

        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.fx_sz = int(math.sqrt(self.num_patch_x))
        self.fz_sz = int(math.sqrt(self.num_patch_z))

        self.decoder = decoder

        self.num_frames = num_frames
        self.num_template = num_template



    def forward(self, template_list=None, search_list=None, template_anno_list=None,
                enc_opt=None, neck_h_state=None, feature=None, mode="encoder"):
        if mode == "encoder":
            return self.forward_encoder(template_list, search_list, template_anno_list)
        elif mode == "neck":
            if hasattr(self, 'neck') and self.neck is not None:  # 检查neck是否存在
                return self.forward_neck(enc_opt, neck_h_state)
            else:
                # 当neck禁用时，直接返回原始特征和空状态
                return {
                    'neck_out': enc_opt,
                    'neck_h_state': None
                }
        elif mode == "decoder":
            return self.forward_decoder(feature)
        else:
            raise ValueError

    def forward_encoder(self, template_list, search_list, template_anno_list):
        # Forward the encoder
        xz = self.encoder(template_list, search_list, template_anno_list)

        # 应用FEM增强
        if self.fem is not None:
            if isinstance(xz, (list, tuple)):
                # 对最后一层进行特征增强
                enhanced = self.fem(xz[-1])
                if isinstance(xz, list):
                    xz[-1] = enhanced
                else:  # tuple
                    xz = list(xz)
                    xz[-1] = enhanced
                    xz = tuple(xz)
            else:
                xz = self.fem(xz)
        return xz

    def forward_neck(self, enc_out, neck_h_state):
        x = enc_out
        xs = x[:, 0:self.num_patch_x]
        x, xs, h = self.neck(x, xs, neck_h_state, self.encoder.body.blocks)
        x = self.encoder.body.fc_norm(x)
        xs = xs + x[:, 0:self.num_patch_x]
        return x, xs, h


    def forward_decoder(self, feature, gt_score_map=None):

        feature = feature[0]
        if self.class_token:
            feature = feature[:,1:self.num_patch_x * self.num_frames+1]
        else:
            feature = feature[:,0:self.num_patch_x * self.num_frames] # (B, HW, C)

        bs, HW, C = feature.size()
        if self.decoder_type in ['CORNER', 'CENTER']:
            feature = feature.permute((0, 2, 1)).contiguous()
            feature = feature.view(bs, C, self.fx_sz, self.fx_sz)
        if self.decoder_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.decoder(feature, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.decoder_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        elif self.decoder_type == "MLP":
            # run the mlp head
            score_map, bbox, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_sutrack(cfg):
    encoder = build_encoder(cfg)
    # 构建FEM模块
    fem = FEM(
        in_channels=cfg.MODEL.FEM.IN_CHANNELS,
        out_channels=cfg.MODEL.FEM.OUT_CHANNELS
    ) if cfg.MODEL.FEM.ENABLED else None

    # 根据配置决定是否构建neck
    neck = build_neck(cfg, encoder) if cfg.MODEL.NECK.ENABLED else None

    decoder = build_decoder(cfg,neck,encoder)

    model = SUTRACK(
        encoder,
        decoder,
        neck,
        fem,
        num_frames = cfg.DATA.SEARCH.NUMBER,
        num_template = cfg.DATA.TEMPLATE.NUMBER,
        decoder_type=cfg.MODEL.DECODER.TYPE,
    )

    # 参数量统计（调试用）
    print("\n=== 模块参数量统计 ===")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"{name.ljust(10)}: {params / 1e6:.2f}M params")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params / 1e6:.2f}M")



    return model
