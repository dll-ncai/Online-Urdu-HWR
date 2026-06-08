from torch import nn

class JointLoss(nn.Module):
    def __init__(
        self,
        blank_id,
        ctc_weight=0.5,
        ce_weight=0.5,
        ignore_index=-100,
        zero_infinity=True
    ):
        super().__init__()

        self.ctc_weight = ctc_weight
        self.ce_weight = ce_weight

        self.ctc_loss = nn.CTCLoss(
            blank=blank_id,
            zero_infinity=zero_infinity
        )

        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index
        )

    def forward(
        self,
        encoder_log_probs,     # (T_enc, B, V)
        encoder_lengths,       # (B,)
        decoder_logits,        # (B, T_dec, V)
        decoder_targets,       # (B, T_dec)
        target_lengths         # (B,)
    ):
        """
        Returns:
            total_loss, ctc_loss, ce_loss
        """

        # -------- CTC LOSS --------
        ctc = self.ctc_loss(
            encoder_log_probs,
            decoder_targets,
            encoder_lengths,
            target_lengths
        )

        # -------- CE LOSS --------
        B, T, V = decoder_logits.shape
        ce = self.ce_loss(
            decoder_logits.reshape(B * T, V),
            decoder_targets.reshape(B * T)
        )

        total = self.ctc_weight * ctc + self.ce_weight * ce

        return total, ctc, ce