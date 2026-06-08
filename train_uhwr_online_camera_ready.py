import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from evaluate import load

cer = load("cer")

from model.joint_multi_modality_model import JointMultiModel
from utils.dataset import ocollate_fn
from utils.dataset import RCOHWRDataset as OHWRDataset
from utils.losses import JointLoss
from utils.decoding import greedy_decode, beam_search_decode
from tokeniser import get_tokenizer

import random
import numpy as np

# SEED = 123
# SEED = 456
SEED = 789
# SEED = 2024

# Fixed dataloader seed for camera-ready: keeps data ordering/shuffling
# identical across runs regardless of the experiment SEED above.
DATALOADER_SEED = 42

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensures deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For reproducibility in some CUDA ops
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    print(f"Seed set to {seed}")



# %%
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def freeze_decoder_layers(model):
    for param in model.transformer_decoder.parameters():
        param.requires_grad = False

def unfreeze_decoder_layers(model):
    for param in model.transformer_decoder.parameters():
        param.requires_grad = True

def compute_loss_weights(epoch,
                          warmup_epochs=5,
                          ramp_epochs=35,
                          ce_max=0.7):
    if epoch < warmup_epochs:
        ce = 0.0
    else:
        progress = min(1.0, (epoch - warmup_epochs) / ramp_epochs)
        ce = ce_max * progress

    ctc = 1.0 - ce
    return ctc, ce

# -------------------------------
# TRAIN ONE EPOCH
# -------------------------------
def train_one_epoch(model, data_loader, optimizer, loss_fn, scaler, tokenizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc="Training", leave=False)

    for batch_idx, batch in enumerate(progress_bar, 1):
        pixel_values = {img_name: batch['pixel_values'][img_name].to(device) for img_name in batch['pixel_values']}
        labels = batch['labels'].to(device)
        tgt = labels.clone()
        tgt[tgt == -100] = tokenizer.pad_token_id
        tgt = tgt[:, :-1] if labels is not None else None
        attention_mask = batch['attention_mask'].to(device)
        tgt_mask = attention_mask[:, :-1] if attention_mask is not None else None

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda"):
            ctc_log_probs, decoder_logits = model(
                pixel_values,
                labels=tgt,
                attention_mask=tgt_mask
            )
            ctc_log_probs = ctc_log_probs.float()

            # ---- CTC targets ----
            ctc_targets = []
            ctc_target_lengths = []

            for i in range(labels.size(0)):
                seq = labels[i]
                seq = seq[
                    (seq != loss_fn.ce_loss.ignore_index) &
                    (seq != tokenizer.bos_token_id) &
                    (seq != tokenizer.eos_token_id)
                ]
                if len(seq) == 0:
                    continue
                ctc_targets.append(seq)
                ctc_target_lengths.append(len(seq))

            if len(ctc_targets) == 0:
                continue  # skip batch safely
            ctc_targets = torch.cat(ctc_targets)
            ctc_target_lengths = torch.tensor(ctc_target_lengths, device=labels.device)

            # ---- CORRECT encoder lengths (T) ----
            T = ctc_log_probs.size(0) if ctc_log_probs.dim() == 3 else ctc_log_probs.size(1)
            encoder_lengths = torch.full(
                (labels.size(0),),
                T,
                dtype=torch.long,
                device=labels.device
            )

            loss, loss_ctc, loss_ce = loss_fn(
                encoder_log_probs=ctc_log_probs,
                encoder_lengths=encoder_lengths,
                decoder_logits=decoder_logits,
                decoder_targets=labels[:, 1:],
                target_lengths=ctc_target_lengths
            )

        scaler.scale(loss).backward()

        # ---- Gradient clipping (Transformer-safe) ----
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        avg_loss = total_loss / batch_idx
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
    return total_loss / len(data_loader)


# -------------------------------
# CER (INTENTIONALLY UNCHANGED)
# -------------------------------
def compute_cer(pred_ids, label_ids, tokenizer, remove_spaces=True):
    """
    pred_ids: (B, T_pred)
    label_ids: (B, T_label)
    """

    # Decode predictions
    pred_strs = tokenizer.batch_decode(
        pred_ids, skip_special_tokens=True
    )

    # Replace -100 before decoding labels
    label_ids = label_ids.clone()
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    label_strs = tokenizer.batch_decode(
        label_ids, skip_special_tokens=True
    )
    # print(pred_strs[0])
    # print(label_strs[0])

    # Optional space normalization (recommended for Urdu OCR)
    if remove_spaces:
        pred_strs  = [s.replace(" ", "") for s in pred_strs]
        label_strs = [s.replace(" ", "") for s in label_strs]


    # Filter empty references (CER requires non-empty refs)
    filtered_preds = []
    filtered_labels = []

    for p, l in zip(pred_strs, label_strs):
        if len(l) > 0:
            filtered_preds.append(" ".join(p))
            filtered_labels.append(" ".join(l))

    # Compute batch CER
    cer_score = cer.compute(
        predictions=filtered_preds,
        references=filtered_labels
    )

    return cer_score


# -------------------------------
# EVALUATION (GREEDY DECODE)
# -------------------------------
def evaluate(model, data_loader, tokenizer, device, loss_fn, loss_calc=True, decode_mode = None):
    model.eval()
    total_loss = 0
    total_cer = 0

    for batch in tqdm(data_loader, desc="Evaluating"):
        pixel_values = {img_name: batch['pixel_values'][img_name].to(device) for img_name in batch['pixel_values']}
        labels = batch['labels'].to(device)
        tgt = labels.clone()
        tgt[tgt == -100] = tokenizer.pad_token_id
        tgt = tgt[:, :-1] if labels is not None else None
        attention_mask = batch['attention_mask'].to(device)
        tgt_mask = attention_mask[:, :-1] if attention_mask is not None else None

        with torch.no_grad():
            if decode_mode == "greedy":
                outputs = greedy_decode(model, pixel_values, 512, tokenizer, device)
            elif decode_mode == "beam_search":
                outputs = beam_search_decode(model, pixel_values, tokenizer, device)

            if loss_calc:
                ctc_log_probs, decoder_logits = model(
                    pixel_values,
                    labels=tgt,
                    attention_mask=tgt_mask
                )

                ctc_targets = []
                ctc_target_lengths = []

                for i in range(labels.size(0)):
                    seq = labels[i]
                    seq = seq[
                        (seq != loss_fn.ce_loss.ignore_index) &
                        (seq != tokenizer.bos_token_id) &
                        (seq != tokenizer.eos_token_id)
                    ]
                    ctc_targets.append(seq)
                    ctc_target_lengths.append(len(seq))

                ctc_targets = torch.cat(ctc_targets)
                ctc_target_lengths = torch.tensor(ctc_target_lengths, device=labels.device)

                T = ctc_log_probs.size(0) if ctc_log_probs.dim() == 3 else ctc_log_probs.size(1)
                encoder_lengths = torch.full(
                    (labels.size(0),),
                    T,
                    dtype=torch.long,
                    device=labels.device
                )

                loss, _, _ = loss_fn(
                    encoder_log_probs=ctc_log_probs,
                    encoder_lengths=encoder_lengths,
                    decoder_logits=decoder_logits,
                    decoder_targets=labels[:, 1:],
                    target_lengths=ctc_target_lengths
                )

                total_loss += loss.item()
            if decode_mode is not None:
                total_cer += compute_cer(outputs, labels, tokenizer)


    return total_loss / len(data_loader), total_cer / len(data_loader)


def save_loss_plot(train_losses, val_losses, output_dir, filename):
    """Persist epoch-wise loss curves for later inspection."""
    if not train_losses:
        return None

    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    if val_losses:
        plt.plot(epochs, val_losses, label="Val Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()

    plot_path = os.path.join(output_dir, filename)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    return plot_path


# -------------------------------
# MAIN (WITH EARLY STOPPING)
# -------------------------------
def main():
    # Set your seed here
    set_seed(SEED)
    # Camera-ready: dataloaders always use the fixed seed 42 so that the
    # data shuffling/ordering is identical across experiment seeds.
    g = torch.Generator()
    g.manual_seed(DATALOADER_SEED)

    ROOT = "C:/AliCode/ICPR/main_repo"
    DATA_ROOT = "C:/AliCode/Datasets/data_online_line_width_alpha"
    os.chdir(ROOT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(f"{DATA_ROOT}/train_leakproof.csv")
    eval_df = pd.read_csv(f"{DATA_ROOT}/test_leakproof.csv")
    # test_df = pd.read_csv(f"{DATA_ROOT}/test_leakproof.csv")
    # train_df = pd.read_csv(f"{DATA_ROOT}/train_extended.csv")
    # eval_df = pd.read_csv(f"{DATA_ROOT}/val_extended.csv")
    # test_df = pd.read_csv(f"{DATA_ROOT}/test_extended.csv")

    # train_df = pd.concat([train_df], ignore_index=True)
    # eval_df = test_df

    tokenizer = get_tokenizer()

    img_feat = "img_stroke"
    # aux_feat = ['img_acceleration', 'img_cos_theta', 'img_curvature', 'img_dt', 'img_dtheta', 'img_dvx', 'img_dvy', 'img_dx', 'img_dy', 'img_pressure', 'img_sin_theta', 'img_speed', 'img_stroke_duration', 'img_stroke_id', 'img_stroke_time', 'img_stroke_time_norm', 'img_theta', 'img_time_norm', 'img_vx', 'img_vy', 'img_x_tilt', 'img_y_tilt']
    # aux_feat = [
    #     'img_dx',
    #     'img_dy',
    #     'img_sin_theta',
    #     'img_cos_theta',
    #     'img_curvature',
    #     'img_pressure',
    # ]
    # aux_feat = [
    #     'img_speed',
    #     'img_sin_theta',
    #     'img_cos_theta',
    #     'img_stroke_time',
    #     'img_pressure',
    # ]
    aux_feat = [
        "img_dx",
        "img_dy",
        "img_sin_theta",
        "img_cos_theta",
        "img_curvature",
        "img_speed",
        "img_acceleration",
        "img_time_norm",
        "img_pressure",
        'img_x_tilt', 'img_y_tilt'
    ]

    train_dataset = OHWRDataset(DATA_ROOT, train_df, tokenizer, aug=True, img_feat=img_feat, aux_feat=aux_feat, cache_dir="train_cache3", training=True,
    aux_dropout_prob=0.25,
    aux_group_dropout_prob=0.15,
    main_dropout_prob = 0.15)
    train_dataset.build_cache()  # Pre-build cache for faster training
    val_dataset   = OHWRDataset(DATA_ROOT, eval_df, tokenizer, aug=False, img_feat=img_feat, aux_feat=aux_feat, cache_dir="test_cache3", training=False)
    val_dataset.build_cache()    # Pre-build cache for faster evaluation

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=ocollate_fn,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=g
    )

    eval_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=ocollate_fn,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=g
    )

    model = JointMultiModel(
        trans_enc_d_model=256,
        trans_enc_nhead=8,
        trans_enc_layers=3,
        trans_enc_ff_dim=1024,
        tokenizer=tokenizer,
        trans_dec_d_model=256,
        trans_dec_nhead=8,
        trans_dec_layers=3,
        trans_dec_n_positions=512,
        freeze_decoder=True,   # 🔥 paper-aligned
        # freeze_decoder=False,  # 🔥 fine-tuning
        # encoder_path="partials_fine_tuned/best_transformer_encoder.pt",
        # decoder_path="partials_fine_tuned/best_transformer_decoder.pt",
        # cnn_encoder_path="partials_fine_tuned/best_cnn_encoder.pt",
        # ctc_head_path="partials_fine_tuned/best_ctc_head.pt",
        encoder_path="partials/best_transformer_encoder_uhwr_icdar.pt",
        decoder_path="partials/best_transformer_decoder_uhwr_icdar.pt",
        cnn_encoder_path="partials/best_cnn_encoder_uhwr_icdar.pt",
        ctc_head_path="partials/best_ctc_head_uhwr_icdar.pt",
        img_feat=img_feat,
        aux_feat=aux_feat,
        freeze_pcnn_encoder = False,
        freeze_tr_encoder = True,
        # fusion_type = "additive"
    ).to(device)
    # model.load_state_dict(torch.load("best_model_online_cross_attention_leakproof_partial_feat_dual_new_new2.pt", map_location=device))

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = JointLoss(
        blank_id=tokenizer.vocab_size,
        ctc_weight=0.8,
        ce_weight=0.2
    )
    scaler = GradScaler()

    max_epochs = 300
    patience = 10
    best_loss = float("inf")
    best_cer = float("inf")
    patience_ctr = 0
    train_loss_history = []
    val_loss_history = []

    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch+1}/{max_epochs}")

        # if epoch == 6:
        #     # unfreeze_decoder_layers(model)
        #     loss_fn.ce_weight = 0.3
        #     loss_fn.ctc_weight = 0.7
        #     # print("🔓 Unfroze decoder layers for fine-tuning.")
        #     print("⚖️ Adjusted loss weights: CTC=0.7, CE=0.3")
        # elif epoch == 26:
        #     loss_fn.ce_weight = 0.7
        #     loss_fn.ctc_weight = 0.3
        #     print("⚖️ Adjusted loss weights: CTC=0.3, CE=0.7")

        # ctc, ce = compute_loss_weights(epoch,
        #                                 warmup_epochs=5,
        #                                 ramp_epochs=35,
        #                                 ce_max=0.7)
        # loss_fn.ctc_weight = ctc
        # loss_fn.ce_weight = ce
        # print(f"⚖️ Adjusted loss weights: CTC={ctc:.2f}, CE={ce:.2f}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, tokenizer, device
        )
        print(f"Train Loss: {train_loss:.4f}")
        train_loss_history.append(train_loss)

        # val_loss, val_cer = evaluate(
        #     model, eval_loader, tokenizer, device, loss_fn, decode_mode="beam_search"
        # )
        val_loss, val_cer = evaluate(
            model, eval_loader, tokenizer, device, loss_fn, decode_mode="greedy"
        )
        print(f"Val Loss: {val_loss:.4f} | Val CER: {val_cer:.4f}")
        val_loss_history.append(val_loss)

        # ---- Early stopping ----
        improved = False

        if val_loss < best_loss - 1e-4:
            improved = True
        elif val_cer < best_cer:
            improved = True

        directory = f"exp_camera_ready_{SEED}"
        os.makedirs(directory, exist_ok=True)
        torch.save(model.state_dict(), f"{directory}/epoch_{epoch}.pt")
        if improved:
            best_loss = min(best_loss, val_loss)
            best_cer = min(best_cer, val_cer)
            patience_ctr = 0
            torch.save(model.state_dict(), f"{directory}/exp_camera_ready_best.pt")
            print("✔ New best model saved")
        else:
            patience_ctr += 1
            print(f"✖ No improvement ({patience_ctr}/{patience})")

        if patience_ctr >= patience:
            print("🛑 Early stopping triggered")
            break

        artifacts_dir = os.path.join(ROOT, "training_artifacts")
        filename = f"loss_curve_online_camera_ready.png"
        plot_path = save_loss_plot(train_loss_history, val_loss_history, artifacts_dir, filename)
        if plot_path:
            print(f"Loss curve saved to {plot_path}")


if __name__ == "__main__":
    main()
