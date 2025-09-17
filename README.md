# Anime Media Generation & Editing Toolkit

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-blue)](https://pytorch.org/)

## [中文版](README.md) | [English](README_EN.md)
## 📖 项目概述
本项目致力于将前沿的开源图像/视频生成与编辑技术与动漫领域深度融合，打造一个集研究、训练与应用为一体的开源平台。我们通过统一的代码框架整合了多种开源图像与视频编辑模型的训练实现，持续追踪最新技术进展并积累优化训练技巧；同时提供精心策划的中等规模动漫图像微调数据集，用于模型训练与验证。此外，项目还包含ComfyUI工作流等多种部署方案，全面展示与应用最新的开源技术成果。

为保持技术前沿性，本项目将积极跟进开源社区动态，基础模型会随主流开源模型的演进不断更新，训练策略也会依据最新研究成果持续优化。因此，代码结构可能会随之调整，不保证版本的完全向前兼容。
### 🚧 TODO
- [✓] Flux Kontext 多条件 训练代码
- [✓] 数据集上传完毕
- [ ] Comfyui 部署代码
- [ ] Flux 强化学习训练代码
- [ ] Flux Kontext 多图片条件编辑数据集
### 📦 安装步骤

1. 克隆项目仓库
```bash
git clone https://github.com/FredChen0001/fc_aniEditor.git
cd fc_aniEditor
```

2. 安装Python依赖
```bash
pip install -r requirements.txt
```
3. 准备数据
- [AniEditor](..%2Fdataset%2FREADME.md)
4. 准备Flux Kontext模型权重
- https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
5. 硬件测试环境
- 8*80G GPUs
6. 运行命令
```bash
cd fc_aniEditor
python -m accelerate.commands.launch --config_file configs/accelerate_config.yaml entrance.py -g configs/kontext_finetune.yaml
```

### 📁 目录结构
```
fc_aniEditor/
├── ComfyUI/ # ComfyUI 部署代码
├── configs/ # 设置实验参数
├── dataset/ # AniEditor 数据集
├── extra_data/ # 模型训练需要的辅助数据
├── logs/ # 训练过程打印的中间结果、保存的权重文件
├── scripts/ # 训练的脚本
├── src/
│   ├── datasets/ # 数据集代码
│   ├── models/ # 各种模型代码
│   └── utils/ # 各种工具代码
├── tried_tricks/ # 训练技巧总结
├── entrance/ # 程序入口
├── LICENSE
└── README.md
```

### 📄 许可证
本项目采用MIT许可证 - 详见LICENSE文件。

### 📞 联系方式
如有问题或建议，请通过以下方式联系：

邮箱: [fredchen0001@163.com]

### 如果您在研究中使用了本代码，请引用以下文献：
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