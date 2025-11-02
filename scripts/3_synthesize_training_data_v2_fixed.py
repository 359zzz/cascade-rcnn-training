import os
import sys
import cv2
import numpy as np
import json
import random
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import multiprocessing as mp
from functools import partial

# 导入配置
sys.path.append('/home/unitree/my_opencv')
from configs.synthesis_config import *

from configs.synthesis_config import (
    SMALL_SIZE_RANGE, MEDIUM_SIZE_RANGE, LARGE_SIZE_RANGE
)

class AnimationStyleSynthesizer:
    """动画风格数据合成器（多进程加速版）"""
    
    def __init__(self, train_val_split=0.8, random_seed=42, num_workers=None):
        """
        初始化合成器
        
        Args:
            train_val_split: 训练集占比（0.8 表示 80% train, 20% val）
            random_seed: 随机种子
            num_workers: 并行进程数（None表示使用CPU核心数）
        """
        self.random_seed = random_seed
        self.train_val_split = train_val_split
        
        # 设置进程数
        if num_workers is None:
            self.num_workers = mp.cpu_count()
        else:
            self.num_workers = num_workers
        
        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        print(f"🎲 随机种子: {random_seed}")
        print(f"📊 Train/Val 划分: {train_val_split:.0%} / {1-train_val_split:.0%}")
        print(f"🚀 并行进程数: {self.num_workers}")
        
        self.load_and_split_backgrounds()
        self.load_and_split_objects()
        
    def load_and_split_backgrounds(self):
        """加载并划分背景（关键修复）"""
        bg_dir = Path(CUSTOM_BG_DIR)
        all_backgrounds = []
        
        extensions = ['.jpg', '.jpeg', '.png']
        
        for ext in extensions:
            all_backgrounds.extend(list(bg_dir.glob(f'*{ext}')))
            all_backgrounds.extend(list(bg_dir.glob(f'**/*{ext}')))
        
        all_backgrounds = list(set(all_backgrounds))
        
        if len(all_backgrounds) == 0:
            raise ValueError(f"错误：在 {CUSTOM_BG_DIR} 中没有找到背景图片！")
        
        # 打乱并划分
        random.shuffle(all_backgrounds)
        split_idx = int(len(all_backgrounds) * self.train_val_split)
        
        # 转换为字符串列表（方便序列化到子进程）
        self.train_backgrounds = [str(p) for p in all_backgrounds[:split_idx]]
        self.val_backgrounds = [str(p) for p in all_backgrounds[split_idx:]]
        
        print(f"✓ 背景图片总数: {len(all_backgrounds)}")
        print(f"  - Train 背景: {len(self.train_backgrounds)}")
        print(f"  - Val 背景: {len(self.val_backgrounds)}")
        
        # 确保 val 至少有一些背景
        if len(self.val_backgrounds) < 10:
            print(f"⚠️  警告：Val 背景太少 ({len(self.val_backgrounds)}), 建议至少 20 张")
    
    def load_and_split_objects(self):
        """加载并划分物体（关键修复）"""
        objects_dir = Path(OBJECTS_DIR)
        
        self.train_objects = {}
        self.val_objects = {}
        
        for class_name in CLASS_WEIGHTS.keys():
            class_dir = objects_dir / class_name
            if not class_dir.exists():
                print(f"⚠️  警告：未找到 {class_name} 目录")
                continue
            
            all_obj_files = list(class_dir.glob('*.png'))
            if len(all_obj_files) == 0:
                print(f"⚠️  警告：{class_name} 目录为空")
                continue
            
            # 打乱并划分
            random.shuffle(all_obj_files)
            split_idx = int(len(all_obj_files) * self.train_val_split)
            
            # 转换为字符串列表
            self.train_objects[class_name] = [str(p) for p in all_obj_files[:split_idx]]
            self.val_objects[class_name] = [str(p) for p in all_obj_files[split_idx:]]
            
            print(f"✓ {class_name}: {len(all_obj_files)} 个物体")
            print(f"  - Train: {len(self.train_objects[class_name])}")
            print(f"  - Val: {len(self.val_objects[class_name])}")
        
        if len(self.train_objects) == 0:
            raise ValueError(f"错误：在 {OBJECTS_DIR} 中没有找到目标物体！")
    
    @staticmethod
    def get_random_background(backgrounds):
        """获取随机背景（静态方法，供子进程调用）"""
        if len(backgrounds) == 0:
            raise ValueError(f"背景为空！")
        
        bg_path = random.choice(backgrounds)
        bg = cv2.imread(bg_path)
        
        if bg is None:
            raise ValueError(f"无法读取背景图片: {bg_path}")
        
        h, w = bg.shape[:2]
        target_h, target_w = TARGET_SIZE
        
        if h < target_h or w < target_w:
            scale = max(target_h / h, target_w / w) * 1.1
            bg = cv2.resize(bg, None, fx=scale, fy=scale)
            h, w = bg.shape[:2]
        
        if h > target_h:
            y = random.randint(0, h - target_h)
        else:
            y = 0
        
        if w > target_w:
            x = random.randint(0, w - target_w)
        else:
            x = 0
        
        bg = bg[y:y+target_h, x:x+target_w]
        
        if bg.shape[:2] != TARGET_SIZE:
            bg = cv2.resize(bg, (target_w, target_h))
        
        return bg
    
    @staticmethod
    def enhance_cartoon_style(img):
        """增强卡通风格"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        sat_factor = random.uniform(*SATURATION_RANGE)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * sat_factor, 0, 255)
        
        bright_factor = random.uniform(*BRIGHTNESS_RANGE)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * bright_factor, 0, 255)
        
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        contrast_factor = random.uniform(*CONTRAST_RANGE)
        img = np.clip(img * contrast_factor, 0, 255).astype(np.uint8)
        
        if random.random() < AUGMENTATION['sharpen']:
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]]) / 1.0
            img = cv2.filter2D(img, -1, kernel)
        
        return img
    
    @staticmethod
    def add_cartoon_outline(obj_img, mask):
        """添加卡通轮廓线"""
        if not ADD_OUTLINE:
            return obj_img
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        result = obj_img.copy()
        cv2.drawContours(result, contours, -1, OUTLINE_COLOR, 
                        OUTLINE_WIDTH, cv2.LINE_AA)
        
        return result
    
    @staticmethod
    def create_simple_shadow(mask, offset=(5, 5)):
        """创建简单扁平阴影"""
        if not ADD_SHADOW:
            return None
        
        shadow = mask.copy()
        h, w = shadow.shape
        
        ox, oy = offset
        M = np.float32([[1, 0, ox], [0, 1, oy]])
        shadow = cv2.warpAffine(shadow, M, (w, h))
        
        shadow = cv2.GaussianBlur(shadow, (SHADOW_BLUR*2+1, SHADOW_BLUR*2+1), 0)
        
        return shadow
    
    @staticmethod
    def place_object_on_background(bg, obj_img, x, y):
        """将物体放置到背景上"""
        obj_h, obj_w = obj_img.shape[:2]
        bg_h, bg_w = bg.shape[:2]
        
        if x < 0 or y < 0 or x + obj_w > bg_w or y + obj_h > bg_h:
            return bg, False
        
        if obj_img.shape[2] == 4:
            obj_rgb = obj_img[:,:,:3]
            mask = obj_img[:,:,3:4] / 255.0
        else:
            obj_rgb = obj_img
            mask = np.ones((obj_h, obj_w, 1), dtype=np.float32)
        
        obj_rgb = AnimationStyleSynthesizer.add_cartoon_outline(
            obj_rgb, (mask[:,:,0]*255).astype(np.uint8))
        
        shadow_mask = AnimationStyleSynthesizer.create_simple_shadow(
            (mask[:,:,0]*255).astype(np.uint8), SHADOW_OFFSET)
        if shadow_mask is not None:
            shadow_mask = shadow_mask.astype(np.float32) / 255.0 * SHADOW_OPACITY
            shadow_mask = shadow_mask[:,:,np.newaxis]
            
            bg_region = bg[y:y+obj_h, x:x+obj_w].astype(np.float32)
            shadow_color = np.array([0, 0, 0], dtype=np.float32)
            bg_region = bg_region * (1 - shadow_mask) + shadow_color * shadow_mask
            bg[y:y+obj_h, x:x+obj_w] = bg_region.astype(np.uint8)
        
        bg_region = bg[y:y+obj_h, x:x+obj_w].astype(np.float32)
        obj_rgb = obj_rgb.astype(np.float32)
        
        blended = bg_region * (1 - mask) + obj_rgb * mask
        bg[y:y+obj_h, x:x+obj_w] = blended.astype(np.uint8)
        
        return bg, True
    
    @staticmethod
    def find_valid_position(bg_w, bg_h, obj_w, obj_h, placed_boxes, max_attempts=50):
        """找到有效的放置位置"""
        margin = MIN_EDGE_DISTANCE
        
        max_obj_w = bg_w - 2 * margin
        max_obj_h = bg_h - 2 * margin
        
        if obj_w > max_obj_w * 0.8 or obj_h > max_obj_h * 0.8:
            return None
        
        for attempt in range(max_attempts):
            if random.random() < LEFT_ZONE_OBJECT_RATIO:
                zone_w = bg_w // 4
                
                x_max = zone_w - obj_w - margin
                if x_max <= margin:
                    x_min = bg_w // 4 + margin
                    x_max = bg_w - obj_w - margin
                else:
                    x_min = margin
                    x_max = max(x_min + 1, x_max)
            else:
                x_min = bg_w // 4 + margin
                x_max = bg_w - obj_w - margin
            
            if x_max <= x_min:
                continue
            
            y_min = margin
            y_max = bg_h - obj_h - margin
            
            if y_max <= y_min:
                continue
            
            x = random.randint(x_min, x_max)
            y = random.randint(y_min, y_max)
            
            new_box = [x, y, x + obj_w, y + obj_h]
            
            if not AnimationStyleSynthesizer.check_overlap(new_box, placed_boxes):
                return (x, y)
        
        return None
    
    @staticmethod
    def check_overlap(new_box, existing_boxes):
        """检查重叠"""
        if not ALLOW_OVERLAP:
            for box in existing_boxes:
                if AnimationStyleSynthesizer.boxes_overlap(new_box, box):
                    return True
            return False
        
        x1, y1, x2, y2 = new_box
        area1 = (x2 - x1) * (y2 - y1)
        
        for box in existing_boxes:
            bx1, by1, bx2, by2 = box
            
            ix1 = max(x1, bx1)
            iy1 = max(y1, by1)
            ix2 = min(x2, bx2)
            iy2 = min(y2, by2)
            
            if ix1 < ix2 and iy1 < iy2:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                overlap_ratio = inter_area / area1
                
                if overlap_ratio > MAX_OVERLAP_RATIO:
                    return True
        
        return False
    
    @staticmethod
    def boxes_overlap(box1, box2):
        """判断两个框是否重叠"""
        x1, y1, x2, y2 = box1
        bx1, by1, bx2, by2 = box2
        
        return not (x2 <= bx1 or x1 >= bx2 or y2 <= by1 or y1 >= by2)
    
    def generate_dataset(self, num_images, output_dir, is_train=True):
        """生成数据集（多进程并行版）"""
        output_dir = Path(output_dir)
        images_dir = output_dir / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n生成 {'训练集' if is_train else '验证集'}: {num_images} 张")
        print(f"使用 {'训练' if is_train else '验证'}专用的背景和物体")
        print(f"并行进程数: {self.num_workers}")
        
        # 准备参数
        backgrounds = self.train_backgrounds if is_train else self.val_backgrounds
        objects_pool = self.train_objects if is_train else self.val_objects
        
        # 创建任务参数列表
        tasks = []
        for i in range(num_images):
            tasks.append({
                'idx': i,
                'backgrounds': backgrounds,
                'objects_pool': objects_pool,
                'seed': self.random_seed + i,  # 每个任务不同的种子
                'images_dir': str(images_dir),
                'is_train': is_train
            })
        
        # 多进程生成
        with mp.Pool(processes=self.num_workers) as pool:
            results = list(tqdm(
                pool.imap(_generate_single_image_worker, tasks),
                total=num_images,
                desc="生成进度"
            ))
        
        # 过滤失败的结果
        results = [r for r in results if r is not None]
        
        print(f"✓ 成功生成 {len(results)} 张图像")
        
        # 整理标注
        images_info = []
        annotations_info = []
        annotation_id = 0
        category_counts = {name: 0 for name in CLASS_WEIGHTS.keys()}
        
        for image_id, result in enumerate(results):
            img_filename, annots = result
            
            # 图像信息
            images_info.append({
                'id': image_id,
                'file_name': img_filename,
                'width': TARGET_SIZE[0],
                'height': TARGET_SIZE[1]
            })
            
            # 标注信息
            for ann in annots:
                x1, y1, x2, y2 = ann['bbox']
                category_name = ann['category']
                category_id = list(CLASS_WEIGHTS.keys()).index(category_name) + 1
                
                annotations_info.append({
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': category_id,
                    'bbox': [x1, y1, x2-x1, y2-y1],
                    'area': ann['area'],
                    'iscrowd': 0
                })
                
                category_counts[category_name] += 1
                annotation_id += 1
        
        # COCO 格式
        coco_format = {
            'images': images_info,
            'annotations': annotations_info,
            'categories': [
                {'id': i+1, 'name': name} 
                for i, name in enumerate(CLASS_WEIGHTS.keys())
            ]
        }
        
        # 保存标注
        ann_file = output_dir.parent / 'annotations' / f"{'train' if is_train else 'val'}_annotations.json"
        ann_file.parent.mkdir(exist_ok=True)
        
        with open(ann_file, 'w') as f:
            json.dump(coco_format, f, indent=2)
        
        # 打印统计
        print(f"✓ 生成完成:")
        print(f"  - 图像: {len(images_info)} 张")
        print(f"  - 标注: {len(annotations_info)} 个")
        print(f"  - 每类物体数量:")
        for name, count in category_counts.items():
            print(f"    • {name}: {count}")


# 全局函数，用于多进程调用
def _generate_single_image_worker(task):
    """工作进程函数"""
    try:
        # 设置进程特定的随机种子
        random.seed(task['seed'])
        np.random.seed(task['seed'])
        
        # 获取背景
        bg = AnimationStyleSynthesizer.get_random_background(task['backgrounds'])
        bg = AnimationStyleSynthesizer.enhance_cartoon_style(bg)
        
        objects_pool = task['objects_pool']
        h, w = bg.shape[:2]
        num_objects = random.randint(MIN_OBJECTS_PER_IMAGE, MAX_OBJECTS_PER_IMAGE)
        
        annotations = []
        placed_boxes = []
        
        attempts = 0
        max_total_attempts = num_objects * 10
        
        while len(annotations) < num_objects and attempts < max_total_attempts:
            attempts += 1
            
            classes = list(objects_pool.keys())
            weights = [CLASS_WEIGHTS[c] for c in classes]
            class_name = random.choices(classes, weights=weights)[0]
            
            obj_path = random.choice(objects_pool[class_name])
            obj_img = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED)
            
            if obj_img is None:
                continue
            
            size_type = random.choices(
                ['small', 'medium', 'large'],
                weights=[SMALL_OBJECT_RATIO, MEDIUM_OBJECT_RATIO, LARGE_OBJECT_RATIO]
            )[0]
            
            if size_type == 'small':
                target_size = random.randint(*SMALL_SIZE_RANGE)
            elif size_type == 'medium':
                target_size = random.randint(*MEDIUM_SIZE_RANGE)
            else:
                target_size = random.randint(*LARGE_SIZE_RANGE)

            obj_h, obj_w = obj_img.shape[:2]
            scale = target_size / max(obj_h, obj_w)
            
            max_dim = min(w, h) // 3
            if target_size > max_dim:
                scale = max_dim / max(obj_h, obj_w)
            
            obj_img = cv2.resize(obj_img, None, fx=scale, fy=scale, 
                                interpolation=cv2.INTER_AREA)
            obj_h, obj_w = obj_img.shape[:2]
            
            if obj_h < 20 or obj_w < 20:
                continue
            
            if random.random() < AUGMENTATION['horizontal_flip']:
                obj_img = cv2.flip(obj_img, 1)
            
            position = AnimationStyleSynthesizer.find_valid_position(
                w, h, obj_w, obj_h, placed_boxes)
            
            if position is None:
                if obj_w > 50 and obj_h > 50:
                    obj_img = cv2.resize(obj_img, None, fx=0.7, fy=0.7)
                    obj_h, obj_w = obj_img.shape[:2]
                    position = AnimationStyleSynthesizer.find_valid_position(
                        w, h, obj_w, obj_h, placed_boxes)
            
            if position is None:
                continue
            
            x, y = position
            
            bg, success = AnimationStyleSynthesizer.place_object_on_background(
                bg, obj_img, x, y)
            
            if success:
                new_box = [x, y, x+obj_w, y+obj_h]
                placed_boxes.append(new_box)
                annotations.append({
                    'bbox': new_box,
                    'category': class_name,
                    'area': obj_w * obj_h
                })
        
        if len(annotations) == 0:
            return None
        
        # 保存图像
        img_filename = f"{'train' if task['is_train'] else 'val'}_{task['idx']:06d}.jpg"
        img_path = Path(task['images_dir']) / img_filename
        cv2.imwrite(str(img_path), bg, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return (img_filename, annotations)
        
    except Exception as e:
        return None


def main():
    print("=" * 70)
    print("动画风格数据合成（多进程并行加速版）")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建合成器（可以指定进程数）
    synthesizer = AnimationStyleSynthesizer(
        train_val_split=0.8,
        random_seed=42,
        num_workers=None  # None = 使用所有CPU核心，或指定数字如 16
    )
    
    # 生成训练集
    synthesizer.generate_dataset(NUM_TRAIN_IMAGES, OUTPUT_TRAIN_DIR, is_train=True)
    
    # 生成验证集
    synthesizer.generate_dataset(NUM_VAL_IMAGES, OUTPUT_VAL_DIR, is_train=False)
    
    print("\n" + "=" * 70)
    print("✓ 数据生成完成！")
    print("=" * 70)
    print(f"训练集: {OUTPUT_TRAIN_DIR}")
    print(f"验证集: {OUTPUT_VAL_DIR}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n⚠️  重要：Train 和 Val 使用了不同的背景和物体，确保独立性！")

if __name__ == '__main__':
    main()
