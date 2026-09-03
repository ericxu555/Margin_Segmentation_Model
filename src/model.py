"""
U-Net segmentation model for tumor segmentation.

This module implements the U-Net architecture using segmentation_models_pytorch
with a ResNet34 encoder pretrained on ImageNet.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class UNetSegmentationModel(nn.Module):
    """
    U-Net model for tumor segmentation with ResNet34 encoder.
    
    The model performs 3-class pixel-level classification:
    - Class 0: Background (including surgical tools)
    - Class 1: Remaining tumor tissue (red)
    - Class 2: Resected tumor tissue (blue)
    
    Architecture:
    - Encoder: ResNet34 pretrained on ImageNet
    - Decoder: U-Net decoder with skip connections
    - Output: 3-channel segmentation mask with softmax activation
    
    Input shape: (B, 3, 512, 512)
    Output shape: (B, 3, 512, 512)
    """
    
    def __init__(self, num_classes: int = 3, pretrained: bool = True, 
                 architecture: str = 'unet', encoder_name: str = 'resnet34'):
        """
        Initialize segmentation model.
        
        Args:
            num_classes: Number of output classes (default: 3)
            pretrained: Whether to use ImageNet pretrained weights for encoder (default: True)
            architecture: Model architecture - 'unet' or 'unetplusplus' (default: 'unet')
            encoder_name: Encoder backbone - 'resnet34', 'resnet50', or 'efficientnet-b2' (default: 'resnet34')
        """
        super().__init__()
        encoder_weights = "imagenet" if pretrained else None
        
        # Select architecture
        if architecture == 'unet':
            model_class = smp.Unet
        elif architecture == 'unetplusplus':
            model_class = smp.UnetPlusPlus
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        self.model = model_class(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation="softmax2d"  # Use softmax2d to avoid dimension warning
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.

        Args:
            x: Input tensor of shape (B, 3, 512, 512)

        Returns:
            Output probabilities of shape (B, 3, 512, 512) after softmax
        """
        return self.model(x)


# ---------------------------------------------------------------------------
# Stage 2: Temporal components
# ---------------------------------------------------------------------------

from typing import Optional, Tuple


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell for spatial-temporal feature learning.

    Applies standard LSTM gating (input / forget / cell / output) using
    2-D convolutions so that spatial structure is preserved in the hidden
    state.  Suitable for insertion at the UNet bottleneck.

    Input shape:  (B, input_dim, H, W)
    Hidden shape: (B, hidden_dim, H, W)
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        """
        Args:
            input_dim:   Number of input channels.
            hidden_dim:  Number of hidden-state channels.
            kernel_size: Spatial kernel size for all gate convolutions (default: 3).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        # One fused convolution for all four gates: i, f, g, o
        self.gates = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Single-step forward pass.

        Args:
            x:     Input tensor (B, input_dim, H, W).
            state: Optional (h, c) tuple from previous timestep.
                   If None, h and c are initialised to zeros.

        Returns:
            h_new:      New hidden state (B, hidden_dim, H, W).
            (h_new, c_new): Updated state tuple.
        """
        B, _, H, W = x.shape

        if state is None:
            h = x.new_zeros(B, self.hidden_dim, H, W)
            c = x.new_zeros(B, self.hidden_dim, H, W)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)            # (B, input_dim+hidden_dim, H, W)
        i, f, g, o = self.gates(combined).chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, (h_new, c_new)


class TemporalUNet(nn.Module):
    """
    UNet with a ConvLSTM cell inserted at the encoder bottleneck.

    Architecture
    ------------
    frame  →  ResNet34 encoder  →  bottleneck (512ch, 16×16)
                                        ↓  proj_in  (1×1, 512→hidden_dim)
                                   ConvLSTMCell (hidden_dim)  ←→ (h, c)
                                        ↓  proj_out (1×1, hidden_dim→512)
                                   UNet decoder (skip connections intact)
                                        ↓
                                   segmentation head  →  softmax probs

    Only the ConvLSTM cell and projection convs are trained during the
    encoder-frozen phase.  After unfreezing, the full model is fine-tuned.

    Input:  (B, 3, 512, 512) per frame
    Output: (B, 3, 512, 512) probability map, updated hidden state tuple
    """

    def __init__(self, base_model: UNetSegmentationModel, hidden_dim: int = 256):
        """
        Args:
            base_model:  A UNetSegmentationModel whose weights have been loaded
                         from the Stage 1 checkpoint.
            hidden_dim:  Number of channels in the ConvLSTM hidden state
                         (default: 256).  A 1×1 conv projects the 512-channel
                         bottleneck down to hidden_dim and back.
        """
        super().__init__()

        # Borrow sub-modules from the smp UNet directly.
        # Assigning nn.Module instances as attributes registers them as
        # sub-modules, so their parameters appear in self.parameters().
        self.encoder = base_model.model.encoder
        self.decoder = base_model.model.decoder
        self.segmentation_head = base_model.model.segmentation_head

        # Bottleneck bridge: 512 → hidden_dim → ConvLSTM → hidden_dim → 512
        self.proj_in = nn.Conv2d(512, hidden_dim, kernel_size=1, bias=False)
        self.convlstm = ConvLSTMCell(hidden_dim, hidden_dim, kernel_size=3)
        self.proj_out = nn.Conv2d(hidden_dim, 512, kernel_size=1, bias=False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for a single frame.

        Args:
            x:            Frame tensor (B, 3, 512, 512).
            hidden_state: Optional (h, c) from the previous frame.
                          Pass None to reset (start of episode / first frame).

        Returns:
            predictions:      Softmax probability map (B, 3, 512, 512).
            new_hidden_state: Updated (h, c) tuple to pass to the next frame.
        """
        # Encoder: returns a list of feature maps at 6 scales
        features = list(self.encoder(x))          # shallow copy so we can replace [-1]

        # Bottleneck temporal bridge
        bottleneck = features[-1]                 # (B, 512, H/32, W/32)
        z = self.proj_in(bottleneck)              # (B, hidden_dim, H/32, W/32)
        h, new_state = self.convlstm(z, hidden_state)
        features[-1] = self.proj_out(h)           # (B, 512, H/32, W/32)

        # Decoder + segmentation head
        decoder_output = self.decoder(features)
        predictions = self.segmentation_head(decoder_output)

        return predictions, new_state

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers
    # ------------------------------------------------------------------

    def freeze_unet(self) -> None:
        """Freeze encoder, decoder, and segmentation head (Phase A)."""
        for p in (
            *self.encoder.parameters(),
            *self.decoder.parameters(),
            *self.segmentation_head.parameters(),
        ):
            p.requires_grad = False

    def unfreeze_unet(self) -> None:
        """Unfreeze encoder, decoder, and segmentation head (Phase B)."""
        for p in (
            *self.encoder.parameters(),
            *self.decoder.parameters(),
            *self.segmentation_head.parameters(),
        ):
            p.requires_grad = True

    # ------------------------------------------------------------------
    # Parameter group helpers (used by TemporalTrainer for dual LR)
    # ------------------------------------------------------------------

    def temporal_parameters(self) -> list:
        """Parameters for the ConvLSTM and projection convolutions only."""
        return (
            list(self.proj_in.parameters())
            + list(self.convlstm.parameters())
            + list(self.proj_out.parameters())
        )

    def unet_parameters(self) -> list:
        """Parameters for the encoder, decoder, and segmentation head."""
        return (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.segmentation_head.parameters())
        )


# ---------------------------------------------------------------------------
# Option C: Temporal Shift Module (TSM) UNet
# ---------------------------------------------------------------------------


class TemporalShiftModule(nn.Module):
    """
    Temporal Shift Module (TSM) — zero-parameter temporal operator.

    Shifts a fraction of channels forward (from t-1) and backward (from t+1)
    along the temporal/batch dimension.  When T consecutive frames are stacked
    as the batch dimension (B=T), this gives every frame implicit access to its
    neighbours without any additional learned parameters.

    Args:
        n_div: Channel fraction to shift in each direction. 1/n_div of
               channels are shifted forward, 1/n_div backward. Default 8
               means 1/8 forward + 1/8 backward = 1/4 of channels are
               temporally mixed.  The remaining 3/4 are unchanged.

    Input / output shape: ``(T, C, H, W)`` — T frames as the batch dim.
    """

    def __init__(self, n_div: int = 8):
        super().__init__()
        self.n_div = n_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, C, H, W = x.shape
        fold = C // self.n_div
        out = x.clone()
        # Forward shift: channels [0:fold] receive context from t-1
        out[1:,  :fold]       = x[:-1, :fold]
        out[0,   :fold]       = 0.0
        # Backward shift: channels [fold:2*fold] receive context from t+1
        out[:-1, fold:2*fold] = x[1:, fold:2*fold]
        out[-1,  fold:2*fold] = 0.0
        return out


class TSMUNet(nn.Module):
    """
    UNet with a Temporal Shift Module inserted at the encoder bottleneck.

    Architecture
    ------------
    T frames  →  (stacked as batch)  →  ResNet34 encoder
                  bottleneck (T, 512, 16, 16)
                        ↓  TemporalShiftModule  (zero extra params)
                  UNet decoder (skip connections intact)
                        ↓
                  segmentation head  →  softmax probs (T, 3, H, W)

    All T frames are processed simultaneously.  The TSM injects temporal
    context at the bottleneck with no learned parameters, making it
    extremely robust to overfitting on small datasets.

    Input:  (T, 3, 512, 512) — T consecutive frames as the batch dim
    Output: (T, 3, 512, 512) — per-frame probability maps
    """

    def __init__(self, base_model: 'UNetSegmentationModel', n_div: int = 8):
        """
        Args:
            base_model: A UNetSegmentationModel whose weights have been loaded
                        from the Stage 1 checkpoint.
            n_div:      Channel fraction for the TSM (default: 8 → 1/8 each
                        direction). Must satisfy 512 % n_div == 0.
        """
        super().__init__()
        self.encoder = base_model.model.encoder
        self.decoder = base_model.model.decoder
        self.segmentation_head = base_model.model.segmentation_head
        self.tsm = TemporalShiftModule(n_div=n_div)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (T, 3, H, W) — T frames stacked in the batch dimension.

        Returns:
            (T, 3, H, W) — per-frame softmax probability maps.
        """
        features = list(self.encoder(x))     # shallow copy; features[-1]: (T, 512, 16, 16)
        features[-1] = self.tsm(features[-1])  # temporal shift at bottleneck
        decoder_output = self.decoder(features)
        return self.segmentation_head(decoder_output)

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers (same interface as TemporalUNet)
    # ------------------------------------------------------------------

    def freeze_unet(self) -> None:
        """Freeze encoder, decoder, and segmentation head."""
        for p in (
            *self.encoder.parameters(),
            *self.decoder.parameters(),
            *self.segmentation_head.parameters(),
        ):
            p.requires_grad = False

    def unfreeze_unet(self) -> None:
        """Unfreeze encoder, decoder, and segmentation head."""
        for p in (
            *self.encoder.parameters(),
            *self.decoder.parameters(),
            *self.segmentation_head.parameters(),
        ):
            p.requires_grad = True

    def tsm_parameters(self) -> list:
        """TSM has no learned parameters — returns empty list."""
        return []

    def unet_parameters(self) -> list:
        """All UNet (encoder + decoder + head) parameters."""
        return (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.segmentation_head.parameters())
        )


# ---------------------------------------------------------------------------
# Option B: Previous-mask guided UNet (4-channel input)
# ---------------------------------------------------------------------------

import torch


class MaskGuidedUNet(nn.Module):
    """
    UNet that accepts a previous-frame segmentation mask as a 4th input channel.

    Architecture
    ------------
    (RGB frame, prev_mask) → cat → (4, H, W) → smp UNet → softmax probs

    The previous mask is encoded as a single float channel:
        background = 0.0,  remaining tumor = 0.5,  resected tumor = 1.0

    Weights are transferred from a Stage 1 UNetSegmentationModel checkpoint.
    The only weight mismatch is the first encoder conv (3 → 4 input channels);
    the new channel is zero-initialised so the model initially ignores the
    previous mask and behaves identically to Stage 1.

    Input:  (B, 3, H, W) frame  +  (B, H, W) prev_mask class indices
    Output: (B, 3, H, W) softmax probability map
    """

    def __init__(
        self,
        base_model: 'UNetSegmentationModel',
        architecture: str = 'unet',
        encoder_name: str = 'resnet34',
    ):
        super().__init__()

        if architecture == 'unet':
            model_class = smp.Unet
        elif architecture == 'unetplusplus':
            model_class = smp.UnetPlusPlus
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Build 4-channel smp model (no pretrained weights — we supply them below)
        self.model = model_class(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=4,
            classes=3,
            activation='softmax2d',
        )

        # Weight surgery: copy all Stage 1 weights that match in shape;
        # for the first encoder conv (only shape mismatch) copy 3 channels
        # and zero-init the new 4th channel.
        base_sd = base_model.model.state_dict()
        new_sd  = self.model.state_dict()
        merged  = {}
        for key, new_w in new_sd.items():
            if key not in base_sd:
                merged[key] = new_w                   # not in Stage 1 — keep init
                continue
            base_w = base_sd[key]
            if new_w.shape == base_w.shape:
                merged[key] = base_w.clone()          # exact match — copy directly
            else:
                # First encoder conv: (out, 3, kH, kW) → (out, 4, kH, kW)
                w = torch.zeros_like(new_w)
                w[:, :base_w.shape[1]] = base_w.clone()   # copy 3 channels
                merged[key] = w                            # 4th channel stays 0
        self.model.load_state_dict(merged)

    def forward(
        self,
        x: torch.Tensor,
        prev_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:         (B, 3, H, W) current frame (normalised, float)
            prev_mask: (B, H, W) previous-frame class indices (0/1/2).
                       Pass torch.zeros(B, H, W) for the first frame of an episode.
        Returns:
            (B, 3, H, W) softmax probability map
        """
        # Encode class indices as a float channel in [0, 1]
        mask_ch = (prev_mask.float() / 2.0).unsqueeze(1)     # (B, 1, H, W)
        return self.model(torch.cat([x, mask_ch], dim=1))     # (B, 4, H, W)
