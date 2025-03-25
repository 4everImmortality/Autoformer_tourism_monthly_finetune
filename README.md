# Autoformer时间序列预测模型微调指南  
# Autoformer Time Series Forecasting Fine-tuning

## 项目概述 / Project Overview

### 中文
本代码库提供对**Autoformer模型**进行时间序列预测任务微调的完整流程，基于Hugging Face生态构建。主要针对`tourism_monthly`（月度旅游数据）数据集，支持：
- 预训练模型加载 ➡ 领域适应微调
- 完整训练评估流水线
- MASE/sMAPE指标计算

### English
This repository provides a complete fine-tuning pipeline for **Autoformer model** on time series forecasting tasks, built on Hugging Face ecosystem. Specifically designed for `tourism_monthly` dataset, supporting:
- Pretrained model loading ➡ Domain-specific fine-tuning
- End-to-end training & evaluation pipeline
- MASE/sMAPE metrics calculation

---

## 核心功能 / Key Features

### 微调流程 / Fine-tuning Process

| 阶段 Stage        | 中文描述                              | English Description                          |
|-------------------|---------------------------------------|----------------------------------------------|
| 数据预处理         | 自动时间特征生成与缺失值处理          | Automated time feature generation & missing value handling |
| 模型初始化         | 加载预训练Autoformer并进行适应性修改  | Load pretrained Autoformer with adaptation    |
| 分布式训练         | 多GPU加速训练支持                     | Multi-GPU accelerated training               |
| 预测生成           | 自动生成24步前瞻预测                  | Auto-generate 24-step ahead forecasts        |
| 结果评估           | 计算行业标准MASE/sMAPE指标            | Compute industry-standard MASE/sMAPE metrics |

---

## 环境配置 / Environment Setup

### 硬件要求 / Hardware
- GPU: NVIDIA CUDA-enabled GPU (推荐/Recommended: ≥16GB VRAM)
- RAM: ≥32GB

### 依赖安装 / Dependencies
```bash
# 基础依赖 Core libraries
pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.36.0 gluonts==0.14.0 accelerate==0.27.0

# 辅助工具 Auxiliary tools
pip install datasets==2.16.0 evaluate==0.4.0 pandas==2.1.0 scipy==1.11.0

