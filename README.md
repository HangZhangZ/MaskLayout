# MaskLayout

MaskLayout is a research-code snapshot for masked floorplan generation/layout completion with three modes:

- image-token only (`img`)
- vector-token only (`vec`)
- hybrid image+vector with graph-aware masking (`hybrid`)

The repository includes model code, preprocessing utilities, notebooks, and several preprocessed `.npy/.npz` training artifacts. It includes all raw data, but the codes are under cleaned up.

## What the project does

High-level pipeline:

1. Parse SwissDwelling-style geometry records into room polygons, adjacency, windows, and boundary/site context.
2. Convert each plan into two aligned representations:
   - vector attributes (`T/S/L/A/R/W`)
   - per-attribute RGBA images (`T/S/L/A/R/W`) and composed visualizations
3. Tokenize attribute images with a VQ-VAE into codebook indices.
4. Train masked transformers to reconstruct masked tokens/attributes.
5. In `hybrid` mode, jointly predict image tokens + vector attributes with graph/attribute/room attention masks.

## Recommended steps (data -> VQ-VAE -> MaskLayout)

Use this order for end-to-end reproduction/training:

1. Download `geometries.csv` from the Swiss Dwelling dataset page on Zenodo: [SwissDwelling (Zenodo)](https://zenodo.org/records/7788422).
2. Run `Extract_SwissD_Data.ipynb` to generate raw dual-modality training assets (vector modality + image assets).
3. Run `VQVAE_Process_Data_55_32.ipynb` to pretrain the VQ-VAE and generate codebook data.
4. Run `main.py` to train MaskLayout.

Availability notes:

- We offer complete vector modality data.
- We offer complete codebook data.
- We offer complete samples of [image data](https://drive.google.com/drive/folders/1X-T7QibzJOgC6UN6m9mlo47GZJHSbf2s?usp=sharing).
- MaskLayout checkpoints are available upon reasonable request.
- We offer pretrained VQ-VAE weights.

## Repository layout

- `main.py`: CLI entrypoint for train / eval / debug sampling.
- `Trainer/`: training loops for `img`, `vec`, and `hybrid` modes plus losses/decoders.
- `Network/MaskLAYOUT.py`: main transformer architectures (image, vector, hybrid, graph-aware hybrid).
- `Network/VQVAE/`: PyTorch VQ-VAE implementation and mask utility functions used by hybrid training.
- `Data_Process/`: SwissDwelling geometry parsing and dual-modality data construction.
- `Extract_SwissD_Data.ipynb`: raw geometry -> images + vector sequences.
- `VQVAE_Process_Data_55_32.ipynb`: TensorFlow VQ-VAE pretraining/data-processing notebook (used to pretrain the VQ-VAE and generate attribute codebooks).
- `Data/`: precomputed tokenized datasets included in this snapshot.

## Included data artifacts

Shapes below are from the current checked-in files:

- `Data/All_codebook_torch.npz`
  - `all`: `(6, 42644, 16, 16)` `int32`
  - `site`: `(42644, 16)` `int32`
- `Data/Site_codebook.npy`
  - `(42644, 25)` `int32`
- `Data/img_mask_complete_3.npz`
  - `T/S/L/A/R/W`: each `(127932, 14, 16)` `uint8`
- `Data/img_mask_partial_3.npz`
  - `T/S/L/A/R/W`: each `(127932, 14, 25)` `uint8`
- `Data/vec_mask_complete_3.npz`
  - `T`: `(127932, 14, 8)`
  - `S`: `(127932, 14, 6)`
  - `L`: `(127932, 14, 2, 7)`
  - `A`: `(127932, 14, 14)`
  - `R`: `(127932, 14, 10, 2, 7)`
  - `W`: `(127932, 14, 2, 7)`
- `Data/vec_mask_partial_3.npz`
  - same as above except `R`: `(127932, 14, 20, 2, 7)`
- `Data/valid_num.npy`
  - `(127932,)` `float64`
- `Data/sqe/swissD_Dual_Modal.npz`
  - `T`: `(42644, 16, 10)` `uint8`
  - `L`: `(42644, 16, 2, 7)` `uint8`
  - `A`: `(42644, 16, 14)` `uint8`
  - `S`: `(42644, 16, 6)` `uint8`
  - `R`: `(42644, 16, 20, 2, 7)` `uint8`
  - `W`: `(42644, 16, 2, 7)` `uint8`
- `Data/sqe/new_R_10.npy`
  - `(42644, 16, 10, 2, 7)` `uint8`
- `Data/img_batch_3.npy` `uint8`
  - `(127932, 16)` `int32`

Notes:

- `127932 = 42644 * 3`; several loaders explicitly triple the site conditioning arrays with `np.concatenate((s, s, s), axis=0)` to match this.
- `42644` is the base plan count used by the checked-in `Data/sqe/*` and codebook artifacts.
- Sequence length is effectively `14` (`--sqe_len` default).
- Attribute channels are `6` (`T, S, L, A, R, W`).

## Representation conventions

### Vector modality (`T/S/L/A/R/W`)

Used in `vec` and `hybrid` training:

- `T`: room type bits / one-hot style representation (training tensor shape uses width `8`)
- `S`: size bits (`6` bits)
- `L`: room location (`2 x 7` bits for x/y)
- `A`: adjacency matrix row (`14` length)
- `R`: room polygon corners (`10` or `20` corners depending on dataset variant) with `(x, y)` bit pairs
- `W`: window location (`2 x 7` bits)

Raw notebook preprocessing starts from larger dimensions (e.g. `T_dim=10`, `R_num=20`) and later downstream training files use reduced/processed variants.

### Image-token modality

Each attribute image is VQ-tokenized into a sequence of discrete code indices.

- `img_mask_complete_3.npz` stores complete token sequences per attribute.
- `img_mask_partial_3.npz` stores partial/masked variants.
- `codebook_size` default is `32`.
- `codebook_shape` default in `main.py` is `16`.

## Model architecture summary

Main model definitions are in `Network/MaskLAYOUT.py`.

Implemented classes include:

- `MaskTransformer_IMG`: image-token masked transformer.
- `MaskTransformer_VEC`: vector-attribute masked transformer.
- `MaskTransformer_HYB_Graph`: joint image+vector transformer with graph/room/attribute masks.
- `TransformerEncoder` + `EncoderLayer`: multi-branch masked attention logic.

Hybrid (`MaskTransformer_HYB_Graph`) behavior:

- encodes image tokens and vector attributes separately
- fuses them in a shared latent sequence
- applies multiple masked attention branches (`all`, `graph`, `attr`, `room`) depending on `mask_mode`
- decodes both image token logits and vector attributes (`T/S/L/A/R/W`)

### `mask_mode` (hybrid attention variants)

Defined in `EncoderLayer.forward()` (`Network/MaskLAYOUT.py`):

- `0`: all + graph + attr + room (all enabled)
- `1`: unmasked global attention + graph + attr + room
- `2`: all + attr + room (no graph branch)
- `3`: all + graph + room (no attr branch)
- `4`: all + graph + attr (no room branch)
- `5`: all four branches but all reuse `all_mask`
- `6`: only unmasked global attention
- `7`: graph + attr + room only (no global branch)
- `8`: all + graph + attr, with graph mask further restricted by room mask

## Training entrypoint (`main.py`)

`main.py` selects a trainer based on `--model_mode`:

- `hybrid` -> `Trainer/vit_hybrid.py`
- `img` -> `Trainer/vit_img.py`
- `vec` -> `Trainer/vit_vec.py`

Common actions:

- training (default): `masklayout.fit()`
- evaluation: `--test-only`
- debug sampling path: `--debug`

### Key CLI args (defaults from `main.py`)

- `--model_mode hybrid|img|vec` (default `hybrid`)
- `--model_size small|big|huge` (default `huge`)
- `--mask_mode` (hybrid attention mode selector)
- `--sqe_len 14`
- `--attri_count 6`
- `--codebook_shape 16`
- `--codebook_size 32`
- `--bsize 64`
- `--epoch 100`
- `--lr 1e-4`
- `--step` (sampling iterations)
- `--sched_mode` (sampling schedule; default `arccos`)
- `--img_loss` (enables extra image reconstruction losses)
- `--resume`
- `--test-only`

### Example commands

Single-GPU hybrid training:

```bash
python main.py --model_mode hybrid --model_size huge --mask_mode 0 --img_loss
```

Hybrid evaluation:

```bash
python main.py --model_mode hybrid --test-only --resume --sched_mode arccos --step 32
```

Multi-GPU (DDP via `torchrun`):

```bash
torchrun --nproc_per_node=4 main.py --model_mode hybrid --model_size huge --mask_mode 0
```

Notes:

- `main.py` auto-detects multiple GPUs and calls NCCL DDP init.
- `LOCAL_RANK` must be set (typically by `torchrun`) for multi-GPU runs.

## Training losses (hybrid)

`Trainer/vit_hybrid.py` combines:

- image token cross-entropy (`out_img` vs codebook tokens)
- MSE losses for vector attributes (`T/S/L/A/R/W`)
- Chamfer distance on decoded room corners (`R`) via `ChamferDistanceLoss_Filter`
- optional composed-image reconstruction loss (`ComposeImgLoss`) when `--img_loss`

Hybrid training also periodically writes sample images to the checkpoint directory:

- `epo{epoch}_img1.jpg`
- `epo{epoch}_vec.png`
- `epo{epoch}_img2.png`

Checkpoint path pattern:

- `args.saved_model/current.pth`

`args.saved_model` is auto-built from `--vit-folder`, `--model_mode`, `--model_size`, `--mask_mode`, and `--img_loss`.

## Data preprocessing pipeline (notebooks)

### 1) `Extract_SwissD_Data.ipynb`

Purpose:

- reads raw geometry CSVs
- parses SwissDwelling floorplans with `Data_Process/SwissD_DataProcess.py`
- writes per-attribute images and sequence tensors

Expected inputs:

- `Data/geometries.csv`
- `Data_Process/room_type.csv`

Key notebook settings:

- `img_res = 128`
- `max_room = 14`
- `T_dim = 10`, `L_dim = 7`, `S_dim = 6`, `R_num = 20`
- `max_boundary_length = 20`
- `num_plan = 45000`

Creates directories such as:

- `Data/img/all`
- `Data/img/site`
- `Data/img/composed`
- `Data/img/composed_all`
- `Data/img/T`, `Data/img/S`, `Data/img/L`, `Data/img/A`, `Data/img/R`, `Data/img/W`
- `Data/sqe`

Saves examples (already given):

- `Data/sqe/swissD_Dual_Modal.npz`
- `Data/sqe/new_R_10.npy`
- `Data/img_batch_3.npy`

### 2) `VQVAE_Process_Data_55_32.ipynb`

Purpose:

- pretrains the TensorFlow VQ-VAE (55_32 setup) and/or loads pretrained encoder/decoder/quantizer weights
- converts per-attribute images to VQ codebook indices
- saves all-attribute codebooks

Expected external assets (pretrained weights are offered):

- `VQ_Pretrained/55_32/vqvae_en/encoder.keras`
- `VQ_Pretrained/55_32/vqvae_de/decoder.keras`
- `VQ_Pretrained/55_32/vqvae_q/quantizer.npy`

Outputs include:

- `Data/All_codebook_torch.npz`

## Environment and dependencies

You will likely need all of the following (even for hybrid mode), because some imported helper modules import TensorFlow at module import time:

- Python 3.9
- `numpy`
- `torch`, `torchvision`, `tensorboard`
- `tensorflow` / `keras`
- `opencv-python`
- `Pillow`
- `shapely`
- `scipy`
- `tqdm`
- `omegaconf`
- `imageio`
- `pandas`, `matplotlib`, `seaborn` (for notebooks)
