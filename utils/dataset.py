import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
from tacobox import Taco
import random
import hashlib
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


class HWRDataset(Dataset):
    def __init__(
        self,
        root,
        df,
        tokenizer,
        input_width=1600,
        input_height=64,
        aug=False,
        taco_aug_frac=0.9,
        max_length: int = 512,
    ):
        self.root = root
        self.df = df.reset_index(drop=True)
        self.input_width = input_width
        self.input_height = input_height
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.mytaco = Taco(
            cp_vertical=0.2,
            cp_horizontal=0.25,
            max_tw_vertical=100,
            min_tw_vertical=10,
            max_tw_horizontal=50,
            min_tw_horizontal=10,
        )

        self.aug = aug
        self.taco_aug_frac = taco_aug_frac

        # Ensure PAD / BOS exist (GPT-2 safety)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df["file_name"][idx]
        text = self.df["text"][idx]

        image_path = os.path.join(self.root, file_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # ---- hard guard: NEVER return None ----
        if image is None or image.size == 0:
            raise ValueError(f"Invalid image: {image_path}")

        pixel_values = self.preprocess(image, self.aug)

        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        # Tokenize WITHOUT special tokens
        tokenized = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - 2,
        )

        input_ids = tokenized["input_ids"]

        # Add BOS / EOS
        input_ids = [bos_id] + input_ids + [eos_id]

        # Attention mask before padding
        attention_mask = [1] * len(input_ids)

        # Pad / truncate
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [pad_id] * pad_len
            attention_mask += [0] * pad_len
        else:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]

        # Labels: PAD → -100 (for CE)
        labels = [
            tok if tok != pad_id else -100
            for tok in input_ids
        ]

        return (
            torch.tensor(pixel_values[None, :, :], dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )

    # -------------------------------------------------
    # IMAGE PREPROCESSING
    # -------------------------------------------------
    def preprocess(self, img, augment=True):
        if augment:
            img = self.apply_taco_augmentations(img)

        # normalize
        img = img.astype(np.float32) / 255.0

        # safety check
        if img.ndim != 2 or img.size == 0:
            raise ValueError("Invalid image after augmentation")

        # (H, W) → (W, H) and flip
        img = img.swapaxes(-2, -1)[..., ::-1]

        target = np.ones(
            (self.input_width, self.input_height), dtype=np.float32
        )

        sx = self.input_width / img.shape[0]
        sy = self.input_height / img.shape[1]
        scale = min(sx, sy)

        new_x = max(1, int(img.shape[0] * scale))
        new_y = max(1, int(img.shape[1] * scale))

        img2 = cv2.resize(img, (new_y, new_x))
        target[:new_x, :new_y] = img2

        return 1.0 - target

    # -------------------------------------------------
    # SAFE TACo AUGMENTATION
    # -------------------------------------------------
    def apply_taco_augmentations(self, input_img):
        h, w = input_img.shape[:2]

        # Too small → skip TACo
        if h < 16 or w < 16:
            return input_img

        if random.random() <= self.taco_aug_frac:
            try:
                augmented = self.mytaco.apply_vertical_taco(
                    input_img,
                    corruption_type="random",
                )
                if augmented is None or augmented.size == 0:
                    return input_img
                return augmented
            except Exception:
                return input_img

        return input_img


def collate_fn(batch):
    src_batch, tgt_batch, attn_mask_batch = [], [], []

    for src_sample, tgt_sample, attn_mask_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
        attn_mask_batch.append(attn_mask_sample)

    return {
        "pixel_values": torch.stack(src_batch),
        "labels": torch.stack(tgt_batch),
        "attention_mask": torch.stack(attn_mask_batch),
    }
    

class OHWRDataset(Dataset):
    def __init__(
        self,
        root,
        df,
        tokenizer,
        input_width=1600,
        input_height=64,
        aug=False,
        taco_aug_frac=0.9,
        max_length: int = 512,
        # img_list=["img_pressure", "img_thickness", "img_x_tilt", "img_y_tilt", "img_height"],
        img_feat = "img_stroke",
        aux_feat = ['img_acceleration', 'img_cos_theta', 'img_curvature', 'img_dt', 'img_dtheta', 'img_dvx', 'img_dvy', 'img_dx', 'img_dy', 'img_pressure', 'img_sin_theta', 'img_speed', 'img_stroke_duration', 'img_stroke_id', 'img_stroke_time', 'img_stroke_time_norm', 'img_theta', 'img_time_norm', 'img_vx', 'img_vy', 'img_x_tilt', 'img_y_tilt'],
    ):
        self.root = root
        self.df = df.reset_index(drop=True)
        self.input_width = input_width
        self.input_height = input_height
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.img_feat = img_feat
        self.aux_feat = aux_feat
        self.mytaco = Taco(
            cp_vertical=0.2,
            cp_horizontal=0.25,
            max_tw_vertical=100,
            min_tw_vertical=10,
            max_tw_horizontal=50,
            min_tw_horizontal=10,
        )

        self.aug = aug
        self.taco_aug_frac = taco_aug_frac

        # Ensure PAD / BOS exist (GPT-2 safety)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        images = {
            img_name: cv2.imread(
                os.path.join(self.root, self.df[img_name][idx]),
                cv2.IMREAD_GRAYSCALE,
            )
            for img_name in [self.img_feat] + self.aux_feat
        }
        
        text = self.df["line"][idx]

        # ---- hard guard: NEVER return None ----
        if any(img is None or img.size == 0 for img in images.values()):
            raise ValueError(f"Invalid image in images: {images}")

        pixel_values = {
            img_name: self.preprocess(img, self.aug)
            for img_name, img in images.items()
        }
        
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        # Tokenize WITHOUT special tokens
        tokenized = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - 2,
        )

        input_ids = tokenized["input_ids"]

        # Add BOS / EOS
        input_ids = [bos_id] + input_ids + [eos_id]

        # Attention mask before padding
        attention_mask = [1] * len(input_ids)

        # Pad / truncate
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [pad_id] * pad_len
            attention_mask += [0] * pad_len
        else:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]

        # Labels: PAD → -100 (for CE)
        labels = [
            tok if tok != pad_id else -100
            for tok in input_ids
        ]

        return (
            {img_name: torch.tensor(pixel_values[img_name][None, :, :], dtype=torch.float32) for img_name in pixel_values},
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )

    # -------------------------------------------------
    # IMAGE PREPROCESSING
    # -------------------------------------------------
    def preprocess(self, img, augment=True):
        if augment:
            img = self.apply_taco_augmentations(img)

        # normalize
        img = img.astype(np.float32) / 255.0

        # safety check
        if img.ndim != 2 or img.size == 0:
            raise ValueError("Invalid image after augmentation")

        # (H, W) → (W, H) and flip
        img = img.swapaxes(-2, -1)[..., ::-1]

        target = np.ones(
            (self.input_width, self.input_height), dtype=np.float32
        )

        sx = self.input_width / img.shape[0]
        sy = self.input_height / img.shape[1]
        scale = min(sx, sy)

        new_x = max(1, int(img.shape[0] * scale))
        new_y = max(1, int(img.shape[1] * scale))

        img2 = cv2.resize(img, (new_y, new_x))
        target[:new_x, :new_y] = img2

        return 1.0 - target

    # -------------------------------------------------
    # SAFE TACo AUGMENTATION
    # -------------------------------------------------
    def apply_taco_augmentations(self, input_img):
        h, w = input_img.shape[:2]

        # Too small → skip TACo
        if h < 16 or w < 16:
            return input_img

        if random.random() <= self.taco_aug_frac:
            try:
                augmented = self.mytaco.apply_vertical_taco(
                    input_img,
                    corruption_type="random",
                )
                if augmented is None or augmented.size == 0:
                    return input_img
                return augmented
            except Exception:
                return input_img

        return input_img
    

def ocollate_fn(batch):
    src_batch, tgt_batch, attn_mask_batch = [], [], []

    for src_sample, tgt_sample, attn_mask_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
        attn_mask_batch.append(attn_mask_sample)

    return {
        "pixel_values": {
            img_name: torch.stack([src[img_name] for src in src_batch])
            for img_name in src_batch[0]
        },
        "labels": torch.stack(tgt_batch),
        "attention_mask": torch.stack(attn_mask_batch),
    }

class COHWRDataset(Dataset):
    def __init__(
        self,
        root,
        df,
        tokenizer,
        cache_dir="aux_cache",
        input_width=1600,
        input_height=64,
        aug=False,
        taco_aug_frac=0.9,
        max_length: int = 512,
        img_feat="img_stroke",
        aux_feat=None,
    ):
        self.root = root
        self.df = df.reset_index(drop=True)
        self.input_width = input_width
        self.input_height = input_height
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.img_feat = img_feat
        self.aug = aug
        self.taco_aug_frac = taco_aug_frac

        if aux_feat is None:
            aux_feat = [
                'img_acceleration', 'img_cos_theta', 'img_curvature', 'img_dt',
                'img_dtheta', 'img_dvx', 'img_dvy', 'img_dx', 'img_dy',
                'img_pressure', 'img_sin_theta', 'img_speed',
                'img_stroke_duration', 'img_stroke_id', 'img_stroke_time',
                'img_stroke_time_norm', 'img_theta', 'img_time_norm',
                'img_vx', 'img_vy', 'img_x_tilt', 'img_y_tilt'
            ]

        self.aux_feat = aux_feat

        # Cache directory (only for AUX channels)
        self.cache_dir = os.path.join(root, cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

        # TACo augmentation
        self.mytaco = Taco(
            cp_vertical=0.2,
            cp_horizontal=0.25,
            max_tw_vertical=100,
            min_tw_vertical=10,
            max_tw_horizontal=50,
            min_tw_horizontal=10,
        )

        # Ensure tokenizer safety
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.df)

    # ==========================================================
    # MAIN GETITEM
    # ==========================================================
    def __getitem__(self, idx):

        # ------------------------------------------------------
        # 1️⃣ MAIN IMAGE (NO CACHE, because AUG may apply)
        # ------------------------------------------------------
        main_path = os.path.join(self.root, self.df[self.img_feat][idx])
        main_img = cv2.imread(main_path, cv2.IMREAD_GRAYSCALE)

        if main_img is None:
            raise ValueError(f"Invalid main image: {main_path}")

        main_processed = self.preprocess(main_img, augment=self.aug)

        # ------------------------------------------------------
        # 2️⃣ AUX CHANNELS (CACHED)
        # ------------------------------------------------------
        if self.aux_feat:
            aux_stack = self.load_or_build_aux_cache(idx)

        # ------------------------------------------------------
        # 3️⃣ Build pixel dictionary
        # ------------------------------------------------------
        pixel_dict = {
            self.img_feat: torch.tensor(
                main_processed[None, :, :], dtype=torch.float32
            )
        }

        for i, name in enumerate(self.aux_feat):
            pixel_dict[name] = torch.tensor(
                aux_stack[i:i+1], dtype=torch.float32
            )

        # ------------------------------------------------------
        # 4️⃣ TOKENIZATION (UNCHANGED)
        # ------------------------------------------------------
        text = self.df["line"][idx]

        tokenized = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - 2,
        )

        input_ids = (
            [self.tokenizer.bos_token_id]
            + tokenized["input_ids"]
            + [self.tokenizer.eos_token_id]
        )

        attention_mask = [1] * len(input_ids)

        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
        else:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]

        labels = [
            tok if tok != self.tokenizer.pad_token_id else -100
            for tok in input_ids
        ]

        return (
            pixel_dict,
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )

    # ==========================================================
    # AUX CACHE HANDLER
    # ==========================================================
    def load_or_build_aux_cache(self, idx):
        stroke_path = self.df["img_stroke"][idx]
        hash_id = hashlib.md5(stroke_path.encode()).hexdigest()
        cache_name = f"{hash_id}_{self.input_width}x{self.input_height}.npy"
        cache_path = os.path.join(self.cache_dir, cache_name)

        # Load if exists
        if os.path.exists(cache_path):
            return np.load(cache_path)

        # Otherwise build cache
        aux_imgs = []

        for name in self.aux_feat:
            path = os.path.join(self.root, self.df[name][idx])
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                raise ValueError(f"Invalid AUX image: {path}")

            # IMPORTANT: no augmentation for AUX
            processed = self.preprocess(img, augment=False)
            aux_imgs.append(processed)

        stacked = np.stack(aux_imgs, axis=0).astype(np.float16)

        np.save(cache_path, stacked, allow_pickle=False)

        return stacked

    # ==========================================================
    # IMAGE PREPROCESSING (UNCHANGED)
    # ==========================================================
    def preprocess(self, img, augment=True):

        if augment:
            img = self.apply_taco_augmentations(img)

        img = img.astype(np.float32) / 255.0

        if img.ndim != 2 or img.size == 0:
            raise ValueError("Invalid image after preprocessing")

        img = img.swapaxes(-2, -1)[..., ::-1]

        target = np.ones(
            (self.input_width, self.input_height), dtype=np.float32
        )

        sx = self.input_width / img.shape[0]
        sy = self.input_height / img.shape[1]
        scale = min(sx, sy)

        new_x = max(1, int(img.shape[0] * scale))
        new_y = max(1, int(img.shape[1] * scale))

        img2 = cv2.resize(img, (new_y, new_x))
        target[:new_x, :new_y] = img2

        return 1.0 - target

    # ==========================================================
    # TACo AUGMENTATION (UNCHANGED)
    # ==========================================================
    def apply_taco_augmentations(self, input_img):

        h, w = input_img.shape[:2]

        if h < 16 or w < 16:
            return input_img

        if random.random() <= self.taco_aug_frac:
            try:
                augmented = self.mytaco.apply_vertical_taco(
                    input_img,
                    corruption_type="random",
                )
                if augmented is None or augmented.size == 0:
                    return input_img
                return augmented
            except Exception:
                return input_img

        return input_img

    def _cache_single(self, idx):
        _ = self[idx]   # triggers cache save inside __getitem__
        return idx

    def build_cache(self, num_workers=None):

        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)

        print(f"Building cache with {num_workers} workers...")
        os.path.exists(self.cache_dir) and os.system(f"rm -rf {self.cache_dir}")

        with Pool(num_workers) as pool:
            list(tqdm(
                pool.imap(self._cache_single, range(len(self))),
                total=len(self)
            ))

        print("Cache building complete.")

import os
import cv2
import random
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


class RCOHWRDataset(Dataset):
    def __init__(
        self,
        root,
        df,
        tokenizer,
        cache_dir="aux_cache",
        input_width=1600,
        input_height=64,
        aug=False,
        taco_aug_frac=0.9,
        max_length: int = 512,
        img_feat="img_stroke",
        aux_feat=None,

        # 🔥 Regularisation Controls
        training=True,
        aux_dropout_prob=0.25,
        aux_group_dropout_prob=0.15,
        aux_noise_std=0.005,
        main_dropout_prob=0.0,  # ✅ default: no dropout for main image
    ):

        self.root = root
        self.df = df.reset_index(drop=True)
        self.input_width = input_width
        self.input_height = input_height
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.img_feat = img_feat
        self.aug = aug
        self.taco_aug_frac = taco_aug_frac

        self.training = training
        self.aux_dropout_prob = aux_dropout_prob
        self.aux_group_dropout_prob = aux_group_dropout_prob
        self.aux_noise_std = aux_noise_std
        self.main_dropout_prob = main_dropout_prob

        if aux_feat is None:
            aux_feat = [
                'img_acceleration', 'img_cos_theta', 'img_curvature', 'img_dt',
                'img_dtheta', 'img_dvx', 'img_dvy', 'img_dx', 'img_dy',
                'img_pressure', 'img_sin_theta', 'img_speed',
                'img_stroke_duration', 'img_stroke_id', 'img_stroke_time',
                'img_stroke_time_norm', 'img_theta', 'img_time_norm',
                'img_vx', 'img_vy', 'img_x_tilt', 'img_y_tilt'
            ]

        self.aux_feat = aux_feat

        # Cache directory
        self.cache_dir = os.path.join(root, cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

        # TACo augmentation
        self.mytaco = Taco(
            cp_vertical=0.2,
            cp_horizontal=0.25,
            max_tw_vertical=100,
            min_tw_vertical=10,
            max_tw_horizontal=50,
            min_tw_horizontal=10,
        )

        # Tokenizer safety
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token

    # ==========================================================
    def __len__(self):
        return len(self.df)

    # ==========================================================
    def __getitem__(self, idx):

        # ------------------------------------------------------
        # 1️⃣ MAIN IMAGE
        # ------------------------------------------------------
        main_path = os.path.join(self.root, self.df[self.img_feat][idx])
        main_img = cv2.imread(main_path, cv2.IMREAD_GRAYSCALE)

        if main_img is None:
            raise ValueError(f"Invalid main image: {main_path}")

        main_processed = self.preprocess(main_img, augment=self.aug)
        main_channel = main_processed[None, :, :].astype(np.float32)

        # ------------------------------------------------------
        # 2️⃣ AUX STACK (cached)
        # ------------------------------------------------------
        aux_stack = None
        if self.aux_feat:
            aux_stack = self.load_or_build_aux_cache(idx)

        # ------------------------------------------------------
        # 3️⃣ DROPOUT LOGIC
        # ------------------------------------------------------
        drop_all_aux = False
        drop_main = False

        if self.training:
            if random.random() < self.aux_group_dropout_prob:
                drop_all_aux = True
            else:
                if random.random() < self.main_dropout_prob:
                    drop_main = True

        # Apply main dropout if selected
        if self.training and drop_main:
            if self.aux_noise_std > 0:
                main_channel = np.random.normal(
                    loc=0.0,
                    scale=self.aux_noise_std,
                    size=main_channel.shape
                ).astype(np.float32)
            else:
                main_channel = np.zeros_like(main_channel)

        pixel_dict = {
            self.img_feat: torch.tensor(main_channel, dtype=torch.float32)
        }

        # ------------------------------------------------------
        # 4️⃣ AUX CHANNEL PROCESSING
        # ------------------------------------------------------
        if aux_stack is not None:
            for i, name in enumerate(self.aux_feat):

                channel = aux_stack[i:i+1].astype(np.float32)

                if self.training:

                    if drop_all_aux:
                        channel = np.zeros_like(channel)

                    else:
                        if random.random() < self.aux_dropout_prob:
                            if self.aux_noise_std > 0:
                                channel = np.random.normal(
                                    loc=0.0,
                                    scale=self.aux_noise_std,
                                    size=channel.shape
                                ).astype(np.float32)
                            else:
                                channel = np.zeros_like(channel)

                pixel_dict[name] = torch.tensor(channel, dtype=torch.float32)

        # ------------------------------------------------------
        # 5️⃣ TOKENIZATION
        # ------------------------------------------------------
        text = self.df["line"][idx]

        tokenized = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - 2,
        )

        input_ids = (
            [self.tokenizer.bos_token_id]
            + tokenized["input_ids"]
            + [self.tokenizer.eos_token_id]
        )

        attention_mask = [1] * len(input_ids)
        pad_len = self.max_length - len(input_ids)

        if pad_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
        else:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]

        labels = [
            tok if tok != self.tokenizer.pad_token_id else -100
            for tok in input_ids
        ]

        return (
            pixel_dict,
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )

    # ==========================================================
    # AUX CACHE HANDLER
    # ==========================================================
    def load_or_build_aux_cache(self, idx):

        stroke_path = self.df["img_stroke"][idx]
        hash_id = hashlib.md5(stroke_path.encode()).hexdigest()
        cache_name = f"{hash_id}_{self.input_width}x{self.input_height}.npy"
        cache_path = os.path.join(self.cache_dir, cache_name)

        if os.path.exists(cache_path):
            return np.load(cache_path)

        aux_imgs = []

        for name in self.aux_feat:
            path = os.path.join(self.root, self.df[name][idx])
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                raise ValueError(f"Invalid AUX image: {path}")

            processed = self.preprocess(img, augment=False)
            aux_imgs.append(processed)

        stacked = np.stack(aux_imgs, axis=0).astype(np.float16)
        np.save(cache_path, stacked, allow_pickle=False)

        return stacked

    # ==========================================================
    # IMAGE PREPROCESSING
    # ==========================================================
    def preprocess(self, img, augment=True):

        if augment:
            img = self.apply_taco_augmentations(img)

        img = img.astype(np.float32) / 255.0
        img = img.swapaxes(-2, -1)[..., ::-1]

        target = np.ones(
            (self.input_width, self.input_height), dtype=np.float32
        )

        sx = self.input_width / img.shape[0]
        sy = self.input_height / img.shape[1]
        scale = min(sx, sy)

        new_x = max(1, int(img.shape[0] * scale))
        new_y = max(1, int(img.shape[1] * scale))

        img2 = cv2.resize(img, (new_y, new_x))
        target[:new_x, :new_y] = img2

        return 1.0 - target

    # ==========================================================
    # TACo AUGMENTATION
    # ==========================================================
    def apply_taco_augmentations(self, input_img):

        h, w = input_img.shape[:2]

        if h < 16 or w < 16:
            return input_img

        if random.random() <= self.taco_aug_frac:
            try:
                augmented = self.mytaco.apply_vertical_taco(
                    input_img,
                    corruption_type="random",
                )
                if augmented is None or augmented.size == 0:
                    return input_img
                return augmented
            except Exception:
                return input_img

        return input_img

    # ==========================================================
    # CACHE BUILDER
    # ==========================================================
    def _cache_single(self, idx):
        _ = self.load_or_build_aux_cache(idx)
        return idx

    def build_cache(self, num_workers=None):

        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)

        print(f"Building cache with {num_workers} workers...")
        os.path.exists(self.cache_dir) and os.system(f"rm -rf {self.cache_dir}")

        with Pool(num_workers) as pool:
            list(tqdm(
                pool.imap(self._cache_single, range(len(self))),
                total=len(self)
            ))

        print("Cache building complete.")