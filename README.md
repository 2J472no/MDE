# ✨ 𝖬𝗈𝗇𝗈𝖼𝗎𝗅𝖺𝗋 𝖣𝖾𝗉𝗍𝗁 𝖤𝗌𝗍𝗂𝗆𝖺𝗍𝗂𝗈𝗇 𝖯𝗋𝗈𝗃𝖾𝖼𝗍 𝗐𝗂𝗍𝗁 𝖬𝖬𝖲𝖾𝗀𝗆𝖾𝗇𝗍𝖺𝗍𝗂𝗈𝗇 𝖢𝗈𝖽𝖾 𝖲𝗍𝗒𝗅𝖾

---

## 📚 Table of Contents
- [🚀 Introduction](#-introduction)
- [💻 Installation](#-installation)
- [⚡ Usage](#-usage)
- [📊 Results](#-results)
- [🙏 Acknowledgements](#-acknowledgements)
- [🤝 Contributing](#-contributing)
- [📝 License](#-license)
- [📚 Citations](#-citations)
- [⏳ Checkpoint](#-checkpoint)

---

## 🚀 Introduction

This project is based on multiple papers, each contributing to different aspects of the project. Below are the key contributions from each paper:

### 📄 [Trap Attention: Monocular Depth Estimation With Manual Traps — CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Ning_Trap_Attention_Monocular_Depth_Estimation_With_Manual_Traps_CVPR_2023_paper.html)
- 🎯 **An efficient attention mechanism**

### 📄 [LR²Depth: Large-Region Aggregation at Low Resolution for Efficient Monocular Depth Estimation — IROS 2025](https://ieeexplore.ieee.org/abstract/document/11246436/)
- 🎯 **A high-speed monocular depth estimation method**

### 📄 [Is Pre-training Applicable to the Decoder for Dense Prediction? (×Net) — ICRA 2026](https://ieeexplore.ieee.org/abstract/document/11246436/)
- 🎯 **A dense prediction method can directly use a baseline model (e.g., ConvNeXt) as the decoder**

### 📄 [MD2E: Modeling Depth-to-Edge Cues for Monocular Metric Depth Estimation — CVPR 2026](https://2j472no.github.io/MD2E/)
- 🎯 **A monocular metric depth estimation method that models depth-to-edge cues**



## 💻 Installation

### 🛠 Prerequisites

Make sure you have the following installed:


### 📥 Clone the Repository

To clone the repository to your local machine, run:

```bash
git clone https://github.com/2J472no/MDE.git
cd MDE
```

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```
> **Note:** MMSegmentation is no longer actively maintained. Using **PyTorch versions higher than those specified in [`requirements.txt`](./requirements.txt)** may introduce compatibility issues or runtime bugs.


## ⚡ Usage

### 🏃‍♂️ Running the Code

To launch training with the provided script:
```bash
bash tools/dist_train.sh <CONFIG> <GPUS> [--other-args]
```

Example:
```bash
bash tools/dist_train.sh configs/LRDepth/1P_L_NYU.py 8
```

MD2E inference:
```bash
python tools/test.py \
  --config configs/MD2E/md2e.py \
  --checkpoint checkpoint.pth \
  --out outputs/md2e_results.pkl
```

MD2E training smoke check:
```bash
python tools/verify_md2e_training.py \
  --config configs/MD2E/md2e.py \
  --checkpoint 
```

MD2E multi-resolution schedule check:
```bash
python tools/verify_md2e_training.py --dry-run-config
```

MD2E multi-resolution training:
```bash
bash tools/dist_train_md2e.sh configs/MD2E/md2e.py 8 \
  --work-dir work_dirs/md2e \
  --options model.decode_head.encoder_checkpoint=checkpoints/dinov2_vitl14_reg4.pth
```

Use `--fixed-resolution` with `tools/train_md2e.py` or `tools/dist_train_md2e.sh` to keep the crop size from `configs/_base_/datasets/md2e.py`.

### ⚙️ Configuration

All config files are located in [`configs/`](./configs). For detailed configuration formats and customization, please refer to [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

For dataset preparation (e.g., **NYU Depth V2** and **KITTI**), please follow the instructions in [Monocular-Depth-Estimation-Toolbox](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox).

For MD2E, point the config at your own data with environment variables:

```bash
export MD2E_DATA_ROOT=<data-root>
export MD2E_IMAGE_DIR=<image-subdirectory>
export MD2E_DEPTH_DIR=<depth-subdirectory>
```

The two file directories are paired by sorted file order. Depth files should contain metric depth values; set `depth_scale` in `configs/_base_/datasets/md2e.py` if your files use a different unit.



---

## 📊 Results

In our paper, we present the following experimental results:

- **[LR²Depth](https://2j472no.github.io/LRDepth/)**
- **[×Net](https://2j472no.github.io/LRDepth/)**
- **[MD2E](https://2j472no.github.io/MD2E/)**

---

## 🙏 Acknowledgements

This project builds upon and is inspired by [MMCV](https://github.com/open-mmlab/mmcv), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), and [timm](https://github.com/huggingface/pytorch-image-models).

---

## ⏳ Checkpoint

- **[LR²Depth](https://drive.google.com/drive/folders/1chwRjjHw_egtt4hIsnoC9dEL_wWUgS5-?usp=sharing)**



---

## 📝 License

This project is licensed under the **MIT License**.

---

## 📚 Citations

If you use this code or project in your research, please cite our paper as follows:

```bash
@InProceedings{Ning_2023_CVPR,
  author    = {Chao Ning and Hongping Gan},
  title     = {Trap Attention: Monocular Depth Estimation With Manual Traps},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023},
  pages     = {5033–5043}
}

@inproceedings{ning2025lr,
  title={LR 2 Depth: Large-Region Aggregation at Low Resolution for Efficient Monocular Depth Estimation},
  author={Ning, Chao and Xuan, Weihao and Gan, Wanshui and Yokoya, Naoto},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={618--625},
  year={2025},
  organization={IEEE}
}

@article{ning2025pre,
  title={Is Pre-training Applicable to the Decoder for Dense Prediction?},
  author={Ning, Chao and Gan, Wanshui and Xuan, Weihao and Yokoya, Naoto},
  journal={arXiv preprint arXiv:2503.07637},
  year={2025}
}

@inproceedings{ning2026md2e,
  title={MD2E: Modeling Depth-to-Edge Cues for Monocular Metric Depth Estimation},
  author={Ning, Chao and Shen, Minghe and Yokoya, Naoto},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}

```

## ✅ TODO
- [ ] ×Net code release
- [ ] Mixed-datasets training
- [ ] Remove mmcv dependency

