
import os
import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='目标检测推理')
    parser.add_argument('--config', 
                       default='/home/unitree/my_opencv/configs/cascade_rcnn_r50_fpn_8gpu.py',
                       help='配置文件路径')
    parser.add_argument('--checkpoint',
                       default='/home/unitree/my_opencv/work_dirs/cascade_rcnn_r50_fpn_8gpu/best_coco_bbox_mAP_epoch_24.pth',
                       help='模型权重文件路径')
    parser.add_argument('--image',
                       required=True,
                       help='输入图片路径')
    parser.add_argument('--output',
                       default='output.jpg',
                       help='输出图片路径')
    parser.add_argument('--score-thr',
                       type=float,
                       default=0.5,
                       help='置信度阈值')
    parser.add_argument('--device',
                       default='cuda:0',
                       help='推理设备')
    return parser.parse_args()

def visualize_detections(image, result, class_names, score_thr=0.5):
    """
    可视化检测结果
    
    Args:
        image: 原始图片
        result: 检测结果
        class_names: 类别名称列表
        score_thr: 置信度阈值
    
    Returns:
        vis_image: 带标注的图片
    """
    vis_image = image.copy()
    
    # 定义颜色（每个类别一个颜色）
    colors = [
        (220, 20, 60),   # 红色
        (119, 11, 32),   # 深红
        (0, 0, 142),     # 蓝色
        (0, 0, 230),     # 深蓝
        (106, 0, 228),   # 紫色
        (0, 60, 100)     # 青色
    ]
    
    # 获取检测结果
    pred_instances = result.pred_instances
    bboxes = pred_instances.bboxes.cpu().numpy()
    labels = pred_instances.labels.cpu().numpy()
    scores = pred_instances.scores.cpu().numpy()
    
    # 统计检测到的目标
    detection_count = {}
    
    # 绘制每个检测框
    for bbox, label, score in zip(bboxes, labels, scores):
        if score < score_thr:
            continue
        
        # 统计
        class_name = class_names[label]
        detection_count[class_name] = detection_count.get(class_name, 0) + 1
        
        # 绘制边界框
        x1, y1, x2, y2 = bbox.astype(int)
        color = colors[label % len(colors)]
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签和置信度
        label_text = f'{class_name}: {score:.2f}'
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        
        # 绘制标签背景
        cv2.rectangle(
            vis_image,
            (x1, y1 - text_height - 10),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # 绘制标签文本
        cv2.putText(
            vis_image,
            label_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
    
    return vis_image, detection_count

def main():
    args = parse_args()
    
    # 注册所有模块
    register_all_modules()
    
    print("=" * 60)
    print("Cascade R-CNN 目标检测推理")
    print("=" * 60)
    print(f"配置文件: {args.config}")
    print(f"模型权重: {args.checkpoint}")
    print(f"输入图片: {args.image}")
    print(f"置信度阈值: {args.score_thr}")
    print("=" * 60)
    
    # 加载模型
    print("\n正在加载模型...")
    model = init_detector(args.config, args.checkpoint, device=args.device)
    print("✓ 模型加载成功")
    
    # 读取图片
    print(f"\n正在读取图片: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print(f"错误: 无法读取图片 {args.image}")
        return
    
    print(f"图片尺寸: {image.shape[1]}×{image.shape[0]}")
    
    # 推理
    print("\n正在进行目标检测...")
    result = inference_detector(model, image)
    print("✓ 检测完成")
    
    # 可视化
    print("\n正在生成可视化结果...")
    class_names = ['object_1', 'object_2', 'object_3', 
                   'object_4', 'object_5', 'object_6']
    
    vis_image, detection_count = visualize_detections(
        image, result, class_names, args.score_thr
    )
    
    # 保存结果
    cv2.imwrite(args.output, vis_image)
    print(f"✓ 结果已保存到: {args.output}")
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("检测结果统计")
    print("=" * 60)
    total = 0
    for class_name, count in sorted(detection_count.items()):
        print(f"{class_name}: {count} 个")
        total += count
    print(f"\n总计: {total} 个目标")
    print("=" * 60)

if __name__ == '__main__':
    main()
