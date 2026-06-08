import copy
import torch
import torch.nn as nn
from .cnn_encoder import CNNEncoder
import os

# ============================================================
# 🔥 Fusion Modules
# ============================================================
def ablation_fusion(x, ablation_type):
    """
    Applies ablation over ALL channels including img_stroke.

    Args:
        x: [B, C_total, H, W]
        ablation_type: string
        feature_names: ordered list including img_stroke first

    Returns:
        Modified tensor x
    """

    feature_names = [
        "img_stroke",
        "img_dx",
        "img_dy",
        "img_sin_theta",
        "img_cos_theta",
        "img_curvature",
        "img_speed",
        "img_acceleration",
        "img_time_norm",
        "img_pressure",
        'img_x_tilt', 'img_y_tilt',
    ]

    if ablation_type is None or ablation_type.lower() == "none":
        return x

    x = x.clone()

    if ablation_type.startswith("Without "):
        feature_name = ablation_type.replace("Without ", "").strip()

        if feature_name not in feature_names:
            raise ValueError(f"{feature_name} not found in features")

        idx = feature_names.index(feature_name)

        x[:, idx:idx+1, :, :] = 0.0

        # print(f"Ablation: WITHOUT {feature_name}")

        return x

    else:
        # ONLY mode
        feature_name = ablation_type.strip()

        if feature_name not in feature_names:
            raise ValueError(f"{feature_name} not found in features")

        keep_idx = feature_names.index(feature_name)

        for i in range(x.shape[1]):
            if i != keep_idx and i != 0:  # Always keep img_stroke (idx 0)
                x[:, i:i+1, :, :] = 0.0

        # print(f"Ablation: ONLY {feature_name}")

        return x

class ChannelFeatureAdaptiveFusion(nn.Module):
    """
    Learns per-channel per-feature weights.

    weights shape: [C, D]
    """

    def __init__(self, num_channels, dim):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_channels, dim))

    def forward(self, branch_outputs):
        x = torch.stack(branch_outputs, dim=0)  # [C, B, T, D]

        alpha = torch.softmax(self.weights, dim=0)  # [C, D]

        alpha = alpha.view(alpha.size(0), 1, 1, alpha.size(1))

        fused = (alpha * x).sum(dim=0)  # [B, T, D]

        return fused

class DynamicAdaptiveFusion(nn.Module):
    """
    Feature-dependent gating fusion.

    alpha = sigmoid(Linear([original || aux]))

    y = alpha * original + (1 - alpha) * aux
    """

    def __init__(self, dim):
        super().__init__()

        self.norm_original = nn.LayerNorm(dim)
        self.norm_aux = nn.LayerNorm(dim)

        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, original_feat, aux_feat):
        """
        original_feat: [B, T, D]
        aux_feat:      [B, T, D]
        """

        # Normalize first (important for stability)
        original_feat = self.norm_original(original_feat)
        aux_feat = self.norm_aux(aux_feat)

        # Concatenate along feature dimension
        concat = torch.cat([original_feat, aux_feat], dim=-1)  # [B, T, 2D]

        # Compute dynamic alpha
        alpha = self.gate(concat)  # [B, T, D]

        # Fuse
        fused = alpha * original_feat + (1 - alpha) * aux_feat
        fused = (original_feat + aux_feat) / 8  # Fallback to simple average if gating fails
        # fused = original_feat

        return fused

# ============================================================
# 🔥 Auxiliary Early-Branch Encoder
# ============================================================

class EarlyBranchAuxiliaryCNN(nn.Module):
    """
    Multi-branch encoder (conv1-3 duplicated)
    conv4-5 shared.
    """

    def __init__(self, in_channels, pretrained_cnn, freeze_shared=True):
        super().__init__()

        self.in_channels = in_channels
        self.cnns = nn.ModuleList()
        self.fusion = ChannelFeatureAdaptiveFusion(num_channels=in_channels, dim=256)

        for _ in range(in_channels):
            cnn_branch = copy.deepcopy(pretrained_cnn)
            self.cnns.append(cnn_branch)

    def forward(self, x):
        """
        x: [B, C, H, W]
        """

        branch_outputs = []
        ablation_type = os.getenv("ABLATION_TYPE", "none")

        for i in range(self.in_channels):
            xi = x[:, i:i+1]
            xi = self.cnns[i](xi)
            branch_outputs.append(xi)

        # x = torch.stack(branch_outputs, dim=0) # [C, B, T, D]
        # x = x.mean(dim=0) # [B, T, D]
        # branch_outputs = ablation_fusion(branch_outputs, ablation_type=ablation_type, aux_feature_names=aux_feat)
        x = self.fusion(branch_outputs) # [B, T, D]
        
        return x


# ============================================================
# 🔥 FINAL FUSED ENCODER
# ============================================================

class FusedCNNEncoder(nn.Module):
    """
    Runs:
        - Original pretrained CNN
        - Auxiliary multi-branch CNN
    Then fuses outputs.
    """

    def __init__(
        self,
        pretrained_cnn,
        aux_in_channels,
        fusion_type="adaptive",
        freeze_shared=True,
    ):
        super().__init__()
        
        self.original_encoder = copy.deepcopy(pretrained_cnn)
        if freeze_shared:
            for param in self.original_encoder.parameters():
                param.requires_grad = False
            self.original_encoder.eval()

        self.aux_encoder = EarlyBranchAuxiliaryCNN(
            in_channels=aux_in_channels,
            pretrained_cnn=pretrained_cnn,
            freeze_shared=freeze_shared,
        )

        embedding_dim = 256  # matches CNN output
            
        self.fusion = DynamicAdaptiveFusion(embedding_dim)

    def forward(self, x):
        """
        x:
            channel 0 -> original image
            channel 1..C -> auxiliary maps

        shape: [B, C_total, H, W]
        """
        x = ablation_fusion(x, ablation_type=os.getenv("ABLATION_TYPE", "none"))
        original = x[:, 0:1]
        aux = x[:, 1:]

        original_feat = self.original_encoder(original)
        aux_feat = self.aux_encoder(aux)

        fused = self.fusion(original_feat, aux_feat)

        return fused


# ============================================================
# 🔥 Factory
# ============================================================

def create_fused_encoder(
    cnn_encoder_path,
    aux_in_channels,
    device="cpu",
    freeze_shared=True,
    fusion_type="adaptive",
    
):

    pretrained = CNNEncoder().to(device)

    state_dict = torch.load(cnn_encoder_path, map_location=device)
    pretrained.load_state_dict(state_dict)
    print("Loaded pretrained original CNN.")

    encoder = FusedCNNEncoder(
        pretrained_cnn=pretrained,
        aux_in_channels=aux_in_channels,
        fusion_type=fusion_type,
        freeze_shared=freeze_shared,
    ).to(device)

    print(
        f"Created fused encoder | Fusion: {fusion_type} | "
        f"Aux branches: {aux_in_channels}"
    )

    return encoder