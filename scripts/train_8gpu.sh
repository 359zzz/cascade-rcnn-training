#!/bin/bash
# 4 GPU 分布式训练脚本

export CUDA_VISIBLE_DEVICES=0,1,2,3

# NCCL 配置
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=WARN

# PyTorch 优化
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

PORT=${PORT:-29500}

echo "=========================================="
echo "🚀 启动 4-GPU 分布式训练"
echo "=========================================="
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "NCCL Interface: ${NCCL_SOCKET_IFNAME}"
echo "Port: $PORT"
echo "Mixed Precision: Enabled"
echo "=========================================="

torchrun \
    --nproc_per_node=4 \
    --master_port=$PORT \
    scripts/train.py \
    --launcher pytorch \
    --amp

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ 训练完成！"
else
    echo "✗ 训练失败，退出码: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
