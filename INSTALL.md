# 安装说明

## 快速开始

```bash
# 1. 创建环境
conda create -n cascade_rcnn python=3.8 -y
conda activate cascade_rcnn

# 2. 安装依赖
bash scripts/setup_environment.sh

# 3. 准备数据
python scripts/1_download_backgrounds.py
# 将物体模板放到 data/raw_templates/
python scripts/prepare_from_manual_templates.py
python scripts/3_synthesize_training_data_v2_fixed.py

# 4. 开始训练
bash scripts/train_8gpu.sh

