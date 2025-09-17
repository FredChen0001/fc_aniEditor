# Anime Media Generation & Editing Toolkit

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-blue)](https://pytorch.org/)

## [中文版](README.md) | [English](README_EN.md)
## 📖 Project Overview
This project is dedicated to deeply integrating cutting-edge open-source image/video generation and editing technologies with the anime domain, creating an open-source platform that integrates research, training, and application. We have integrated the training implementations of various open-source image and video editing models through a unified code framework, continuously tracking the latest technological advancements and accumulating optimized training techniques. Additionally, the project provides a carefully curated medium-scale anime image fine-tuning dataset for model training and validation. Moreover, it includes various deployment solutions such as ComfyUI workflows to comprehensively demonstrate and apply the latest open-source technological achievements.

To maintain technological cutting-edge, this project will actively follow the dynamics of the open-source community. The base models will be continuously updated with the evolution of mainstream open-source models, and training strategies will be continuously optimized based on the latest research results. Therefore, the code structure may be adjusted accordingly, and complete forward compatibility of versions is not guaranteed.

### 🚧 TODO
- [✓] Flux Kontext multi-condition training code
- [✓] Dataset upload completed
- [ ] ComfyUI deployment code
- [ ] Flux RL code
### 📦 Installation Steps

1. Clone the repository
```bash
git clone https://github.com/FredChen0001/fc_aniEditor.git
cd fc_aniEditor
```

2. Install Python dependencies
```bash
pip install -r requirements.txt
```

3. Prepare data
- [AniEditor](..%2Fdataset%2FREADME.md)

4. Prepare Flux Kontext model weights
- https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev

5. Hardware Environment
- 8×80GB GPUs
6. run command
```bash
cd fc_aniEditor
python -m accelerate.commands.launch --config_file configs/accelerate_config.yaml entrance.py -g configs/kontext_finetune.yaml
```
### 📁 Directory Structure
```
fc_aniEditor/
├── ComfyUI/ # ComfyUI deployment code
├── configs/ # Experimental parameter settings
├── dataset/ # AniEditor dataset
├── extra_data/ # Auxiliary data required for model training
├── logs/ # Intermediate results printed during training, saved weight files
├── scripts/ # Training scripts
├── src/
│   ├── datasets/ # Dataset code
│   ├── models/ # Various model codes
│   └── utils/ # Various utility codes
├── tried_tricks/ # Summary of training techniques
├── entrance/ # Program entry
├── LICENSE
└── README.md
```

### 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

### 📞 Contact
For questions or suggestions, please contact us via:

Email: fredchen0001@163.com

### If you use this code in your research, please cite the following papers:
```
@misc{AniEditor2025,
  author       = {FredChen0001},
  title        = {{AniEditor: A Comprehensive Dataset for Animation Editing Research}},
  howpublished = {\url{https://github.com/FredChen0001/fc_aniEditor}},
  year         = {2025},
  note         = {GitHub repository},
  publisher    = {GitHub},
}
```