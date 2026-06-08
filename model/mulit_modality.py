import torch
import torch.nn as nn
from .multi_model_new9 import create_fused_encoder


class MultiModalCNNEncoder3(nn.Module):
    """Multi-modal CNN encoder used by JointMultiModel.

    Stacks the ink/stroke image and the auxiliary feature maps into a single
    [B, 1 + len(aux_feat), H, W] tensor and feeds it to the fused encoder
    (original pretrained CNN on the ink channel + per-aux-channel branches,
    then adaptive fusion). See model/multi_model_new9.py.
    """

    def __init__(
        self,
        cnn_encoder_path=None,
        img_feat="img_stroke",
        aux_feat=[
            'img_acceleration', 'img_cos_theta', 'img_curvature',
            'img_dt', 'img_dtheta', 'img_dvx', 'img_dvy',
            'img_dx', 'img_dy', 'img_pressure',
            'img_sin_theta', 'img_speed',
            'img_stroke_duration', 'img_stroke_id',
            'img_stroke_time', 'img_stroke_time_norm',
            'img_theta', 'img_time_norm',
            'img_vx', 'img_vy',
            'img_x_tilt', 'img_y_tilt',
        ],
        freeze_pcnn_encoder=False,
        device='cpu',
        fusion_type="adaptive"
    ):
        super().__init__()

        self.img_feat = img_feat
        self.aux_feat = aux_feat
        self.device = device
        self.freeze_pcnn_encoder = freeze_pcnn_encoder
        self.cnn_encoder = create_fused_encoder(
            cnn_encoder_path=cnn_encoder_path,
            aux_in_channels=len(self.aux_feat),
            device=device,
            freeze_shared=freeze_pcnn_encoder,
            fusion_type=fusion_type
        )

    def forward(self, pixel_values):
        # Stack ink + auxiliary modalities -> [B, 1 + len(aux_feat), H, W]
        aux_input = torch.cat(
            [pixel_values[key] for key in [self.img_feat] + self.aux_feat],
            dim=1
        )
        f_aux = self.cnn_encoder(aux_input)  # [B, T, 256]
        return f_aux
