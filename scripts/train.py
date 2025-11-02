import os
import sys
import argparse
from mmdet.utils import register_all_modules
from mmengine.config import Config
from mmengine.runner import Runner

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', 
                       default='/home/unitree/my_opencv/configs/cascade_rcnn_r50_fpn_8gpu.py',
                       help='train config file path')
    parser.add_argument('--launcher', 
                       choices=['none', 'pytorch', 'slurm', 'mpi'],
                       default='none',
                       help='job launcher')
    parser.add_argument('--amp', action='store_true',
                       help='enable automatic mixed precision training')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # 从环境变量读取 local_rank
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # ============ 性能优化环境变量 ============
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
    
    # 注册所有模块
    register_all_modules()
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    
    # 设置分布式训练
    if args.launcher != 'none':
        cfg.launcher = args.launcher
    
    # ============ 关键加速配置 ============
    
    # 1. 混合精度训练（最重要！2-3倍加速）
    if args.amp:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'
        if local_rank == 0:
            print("✓ 启用混合精度训练 (AMP)")
    
    # 2. 启用 cudnn benchmark（必须！）
    cfg.env_cfg.cudnn_benchmark = True
    
    # 3. 数据加载优化
    cfg.train_dataloader.num_workers = 8  # 增加到 8
    cfg.train_dataloader.persistent_workers = True
    cfg.train_dataloader.pin_memory = True
    
    # 添加 prefetch_factor（如果支持）
    if not hasattr(cfg.train_dataloader, 'prefetch_factor'):
        cfg.train_dataloader.prefetch_factor = 4
    
    # 4. 验证集加载优化
    cfg.val_dataloader.num_workers = 4
    cfg.val_dataloader.persistent_workers = True
    cfg.val_dataloader.pin_memory = True
    
    # 5. 减少验证频率（可选）
    if hasattr(cfg.train_cfg, 'val_interval'):
        cfg.train_cfg.val_interval = 4  # 从 2 改到 4
    
    # 6. 减少日志频率
    if hasattr(cfg.default_hooks, 'logger'):
        cfg.default_hooks.logger.interval = 100  # 从 50 改到 100
    
    # 7. checkpoint 频率
    if hasattr(cfg.default_hooks, 'checkpoint'):
        cfg.default_hooks.checkpoint.interval = 4  # 从 2 改到 4
    
    # 创建工作目录
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    # 只在主进程打印信息
    if local_rank == 0:
        print("=" * 60)
        print("🚀 开始训练 Cascade R-CNN 模型（加速版）")
        print("=" * 60)
        print(f"配置文件: {args.config}")
        print(f"工作目录: {cfg.work_dir}")
        print(f"训练轮数: {cfg.train_cfg.max_epochs}")
        print(f"Batch size per GPU: {cfg.train_dataloader.batch_size}")
        print(f"Total batch size: {cfg.train_dataloader.batch_size * int(os.environ.get('WORLD_SIZE', 1))}")
        print(f"Workers per GPU: {cfg.train_dataloader.num_workers}")
        print(f"Validation interval: {cfg.train_cfg.val_interval} epochs")
        print(f"Mixed precision: {args.amp}")
        print(f"cuDNN benchmark: {cfg.env_cfg.cudnn_benchmark}")
        print("=" * 60)
    
    # 构建 runner
    runner = Runner.from_cfg(cfg)
    
    # 开始训练
    runner.train()
    
    if local_rank == 0:
        print("\n" + "=" * 60)
        print("✓ 训练完成！")
        print("=" * 60)
        print(f"模型保存在: {cfg.work_dir}")

if __name__ == '__main__':
    main()