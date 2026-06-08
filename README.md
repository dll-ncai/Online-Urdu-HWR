# Online Urdu Handwriting Recognition

Code release for our work on **online Urdu handwriting text recognition (HTR)**.
The model combines a per-channel CNN feature extractor over rendered stroke /
temporal-feature images with a Transformer encoder and a Transformer decoder, and
is trained with a joint CTC + cross-entropy objective.

This repository contains the **camera-ready** training and evaluation code used to
produce the reported results. Datasets and trained checkpoints are **not** included
(see [Data & checkpoints](#data--checkpoints)).

## Repository layout

```
model/                       Model architecture
  joint_multi_modality_model.py   Online model (ink + N auxiliary channels)
  joint_model.py                  Fine-tune model (stroke-only)
  mulit_modality.py, multi_model_new9.py, mulit_model_new.py
  cnn_encoder.py, transformer_encoder.py, transformer_decoder.py, ctc_head.py
utils/
  dataset.py                  Datasets + collate fns (online & offline)
  decoding.py                 Greedy / beam-search decoding
  losses.py                   Joint CTC + CE loss
tokeniser.py                  GPT-2 byte-level tokenizer (vocabs/ved/)
vocabs/ved/                   Tokenizer vocab + merges
DatasetProcessing/            Data prep: rendering & temporal-feature scripts

Training / evaluation
  train_uhwr_online_camera_ready.py        Online model, single seed
  train_uhwr_online_camera_ready_5runs.py  Online model, 5 seeds
  train_uhwr_fine_tune_camera_ready.py     Fine-tune model, single seed
  train_uhwr_fine_tune_camera_ready_5runs.py  Fine-tune model, 5 seeds
  train_single_channel_camera_ready.ipynb  Per-channel "ink + one aux" sweep
  Calc_cer_camera_ready.ipynb              CER evaluation + channel ablations
  train_uhwr_online.py, train_uhwr.py      Base modules (provide evaluate())

cer_camera_ready_pairs.csv, cer_camera_ready_ablations.csv   Reported CER tables
```

## The two models

- **Online (multi-modal):** `JointMultiModel` consumes the ink/stroke image plus up
  to 11 auxiliary temporal-feature channels (`img_dx, img_dy, img_sin_theta,
  img_cos_theta, img_curvature, img_speed, img_acceleration, img_time_norm,
  img_pressure, img_x_tilt, img_y_tilt`). The CNN encoder is built with
  `aux_in_channels = len(aux_feat)`, so any subset of channels works.
- **Fine-tune (stroke-only):** `JointModel` uses only the stroke image
  (`aux_feat=[]`).

The single-channel notebook trains a fresh `JointMultiModel` for each
**(seed, aux channel)** pair using only **ink + that one channel** — isolating each
channel's contribution with a model actually optimized for it (5 seeds × 11
channels). `Calc_cer_camera_ready.ipynb` reproduces the inference-time channel
ablations by masking channels at evaluation.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.11 is recommended. Install the PyTorch build that matches your CUDA/CPU.

## Configuration (important)

The training scripts and notebooks use **absolute paths** that you must edit for
your machine. In each script's `main()` (and at the top of the notebooks), set:

```python
ROOT = "/path/to/this/repo"
DATA_ROOT = "/path/to/your/dataset"   # contains train/val/test_leakproof.csv
```

The dataset is expected to provide `train_leakproof.csv`, `val_leakproof.csv`, and
`test_leakproof.csv`, plus the rendered feature images referenced therein. See
`DatasetProcessing/` for how these images and temporal features are generated.

## Data & checkpoints

Not distributed in this repository:
- **Datasets** and rendered feature images / caches (`*_cache3/`).
- **Pretrained initialization weights** under `partials/` (CNN encoder,
  Transformer encoder/decoder, CTC head) that the trainers load to initialize the
  model.
- **Trained checkpoints** and experiment outputs (`runs_*/`, `*.pt`).

To run end-to-end you need to supply your dataset and the `partials/` init weights
(paths are referenced in the trainer scripts). Please contact the authors for
access to the pretrained weights.

## Training

```bash
# Online model across 5 seeds -> runs_online_camera_ready/seed_*/
python train_uhwr_online_camera_ready_5runs.py

# Fine-tune (stroke-only) across 5 seeds -> runs_finetune_camera_ready/seed_*/
python train_uhwr_fine_tune_camera_ready_5runs.py
```

Per-channel sweep: open `train_single_channel_camera_ready.ipynb`. Use the
`CHANNELS_TO_RUN` / `SEEDS_TO_RUN` knobs to run in batches; completed runs (those
with a `best.pt`) are skipped. **Run the smoke test first** (one channel, one seed,
`MAX_EPOCHS=1`) before launching the full 55-run sweep.

## Evaluation

Open `Calc_cer_camera_ready.ipynb`. For each seed it selects the lowest-CER
checkpoint within a scan window (last 40% of epochs, excluding the final 9, to avoid
the early-stopping tail), then reports per-seed and mean CER plus channel-addition
and ink-only ablations. Results are written to `cer_camera_ready_pairs.csv` and
`cer_camera_ready_ablations.csv` (included here as the reported tables).

## Reproducibility

All trainers fix a per-run experiment seed and a **fixed dataloader seed** so that
data ordering/shuffling is identical across runs. Seeds used: `42, 123, 456, 789,
2024`.

## Citation

If you use this code, please cite our paper. (Citation details to be added.)
