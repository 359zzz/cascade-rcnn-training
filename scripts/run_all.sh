#!/bin/bash
# ===========================================
# scripts/run_all.sh
# 一键运行所有步骤
# ===========================================

set -e  # 遇到错误立即退出

echo "======================================"
echo "模板预处理 + 训练数据生成 完整流程"
echo "======================================"

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cascade_rcnn

# 设置工作目录
cd /home/unitree/my_opencv

echo ""
echo "[步骤 1/4] 检查环境和依赖..."
python -c "import cv2, numpy, PIL, rembg, albumentations; print('✓ 所有依赖已安装')"

echo ""
echo "[步骤 2/4] 去除模板背景..."
python scripts/1_remove_background.py

echo ""
echo "[步骤 3/4] 生成多尺度模板库..."
python scripts/2_generate_template_library.py

echo ""
echo "[步骤 4/4] 合成训练数据..."
python scripts/3_synthesize_training_data.py

echo ""
echo "======================================"
echo "✓ 所有步骤完成！"
echo "======================================"
echo ""
echo "生成的数据位置:"
echo "  - 处理后的模板: data/processed_templates/"
echo "  - 训练数据: data/synthetic_train/"
echo "  - 验证数据: data/synthetic_val/"
echo "  - 标注文件: data/annotations/"
echo ""
echo "下一步: 使用生成的数据训练 Cascade R-CNN 模型"

