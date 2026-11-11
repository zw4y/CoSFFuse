# CoSFFuse

**CoSFFuse: Spatial-Dominant Guidance Guiding Auxiliary Collaborative Frequency Enhancement for Infrared--Visible Image Fusion**

This repository provides the complete implementation and trained weights for reproducing the experiments in the paper, including the network architecture (`modules.py`), the training script (`train.py`), fusion inference (`fuse.py`), quantitative evaluation (`val.py`), and the data-processing utilities.

## Requirements

- Python 3.8+
- PyTorch (tested with 2.x) and torchvision
- Kornia
- OpenCV, NumPy, tqdm, matplotlib
- PyYAML
- wandb (required by `train.py`; set `wandb_mode: offline` in `configs/cfg.yaml` to run without an account)

```bash
pip install torch torchvision kornia opencv-python numpy tqdm matplotlib pyyaml wandb
```

## Datasets

All datasets used in this work are publicly available and are **not** included in this repository:

- **MSRS** (training and in-distribution testing): <https://github.com/Linfeng-Tang/MSRS>
- **M3FD**: <https://github.com/dlut-dimt/TarDAL>
- **LLVIP**: <https://github.com/bupt-ai-cz/LLVIP>

Expected layout (matching the default configuration in `configs/cfg.yaml`):

```
./MSRS/                     # place the downloaded dataset inside the repository root
  train/ir/*.png
  train/vi/*.png
  test/ir/*.png
  test/vi/*.png
```

For cross-dataset evaluation, arrange the M3FD and LLVIP test pairs as follows and set the input and output paths in `fuse.py` and `val.py` for the dataset you want to test:

```
./M3FD_50pairs/test/ir/*.png    ./M3FD_50pairs/test/vi/*.png
./LLVIP_50pairs/test/ir/*.jpg   ./LLVIP_50pairs/test/vi/*.jpg
```

The model is trained on the MSRS official split (1083 training pairs) and applied directly, without fine-tuning, to the M3FD and LLVIP test pairs. Keeping the datasets inside the repository root places the data and the code in one location.

## Trained weights

`models/lastest.pth` is the trained MSRS model used to produce the fusion results reported in the paper. It is the default checkpoint in the commands below.

## Reproducing the experiments

Run the following commands from the repository root. Both `fuse.py` and `val.py` use the MSRS test split by default.

### 1. Training

```bash
python train.py
```

Training configuration (`configs/cfg.yaml`): MSRS official split (1083 training pairs), input size 320x320, 200 epochs, batch size 4, Adam optimizer with initial learning rate 5e-4, automatic mixed precision and gradient accumulation. If wandb is used, set `WANDB_API_KEY` or pass `--auth <key>`; to run without wandb, set `wandb_mode: offline` in the config.

### 2. Fusion inference

```bash
python fuse.py
```

### 3. Quantitative evaluation

```bash
python val.py
```

`val.py` computes entropy (EN), spatial frequency (SF), mutual information (MI), visual information fidelity (VIF), the Xydeas--Petrovic edge-based fusion quality metric (Q_abf), and average gradient (AG), using the same unified implementation for all compared methods.

## Acknowledgement

`utils/u2net.py` and `utils/u2netp.pth` are from the [U2-Net](https://github.com/xuebinqin/U-2-Net) project (Apache-2.0).
