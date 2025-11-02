import cv2
import numpy as np
from pathlib import Path
import random

class ObjectPreparer:
    def __init__(self):
        self.template_dir = Path('data/raw_templates')
        self.output_dir = Path('data/objects')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ============ 可调参数 ============
        # 变体生成数量
        self.num_variants = 200
        
        # 线条干扰参数
        self.line_interference_ratio = 0.4  # 30%的变体添加线条
        self.num_lines_per_image = 1        # 每个变体的线条数量
        self.line_thickness_range = (1, 1)  # 线条粗细范围
        self.line_style = 'solid'           # 'solid' 或 'dashed'
        
        # 线条颜色分布
        self.line_color_distribution = {
            'dark': 0.8,      # 70% 深色
            'color': 0.1,     # 20% 彩色
            'transparent': 0.1 # 10% 半透明
        }
        
        # 线条长度（相对于对角线）
        self.line_length_range = (0.9, 1.8)
        
        print("=" * 60)
        print("ObjectPreparer 配置：")
        print(f"  变体数量: {self.num_variants}")
        print(f"  线条干扰比例: {self.line_interference_ratio * 100:.0f}%")
        print(f"  每图线条数: {self.num_lines_per_image}")
        print(f"  线条样式: {self.line_style}")
        print("=" * 60)
        
    def load_template(self, template_path):
        """加载模板图片（已有alpha通道）"""
        img = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"无法读取: {template_path}")
        
        # 确保有alpha通道
        if img.shape[2] == 3:
            print(f"⚠️  {template_path.name} 没有alpha通道，添加默认alpha")
            # 基于亮度创建alpha
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, alpha = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            img[:,:,3] = alpha
        
        return img
    
    def crop_to_content(self, img, margin=15):
        """裁剪到内容区域"""
        alpha = img[:,:,3]
        coords = cv2.findNonZero(alpha)
        if coords is None:
            return img
        
        x, y, w, h = cv2.boundingRect(coords)
        
        # 添加边距
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img.shape[1] - x, w + 2*margin)
        h = min(img.shape[0] - y, h + 2*margin)
        
        return img[y:y+h, x:x+w]
    
    def enhance_edges(self, img):
        """轻微增强边缘（减少羽化，保持清晰）"""
        alpha = img[:,:,3].astype(float) / 255.0
        
        # 非常轻微的羽化
        alpha_blur = cv2.GaussianBlur(alpha, (3, 3), 0.3)
        
        # 混合比例调整：更偏向原始清晰边缘
        alpha_final = 0.9 * alpha + 0.1 * alpha_blur
        alpha_final = np.clip(alpha_final * 255, 0, 255).astype(np.uint8)
        
        img[:,:,3] = alpha_final
        return img
    
    def generate_line_color(self):
        """生成线条颜色"""
        color_type = random.choices(
            ['dark', 'color', 'transparent'],
            weights=[
                self.line_color_distribution['dark'],
                self.line_color_distribution['color'],
                self.line_color_distribution['transparent']
            ]
        )[0]
        
        if color_type == 'dark':
            # 深色系
            colors = [
                (0, 0, 0),        # 黑色
                (40, 40, 40),     # 深灰
                (60, 30, 0),      # 深棕
                (0, 0, 60),       # 深蓝
            ]
            return random.choice(colors), 1.0
        
        elif color_type == 'color':
            # 彩色
            colors = [
                (0, 0, 255),      # 红色
                (0, 255, 0),      # 绿色
                (255, 0, 0),      # 蓝色
                (0, 255, 255),    # 黄色
            ]
            return random.choice(colors), 1.0
        
        else:  # transparent
            # 半透明深色
            color = (random.randint(0, 60), random.randint(0, 60), random.randint(0, 60))
            alpha = random.uniform(0.6, 0.9)
            return color, alpha
    
    def generate_random_line_layer(self, width, height):
        """
        生成随机线条图层（透明背景）
        
        返回: 包含线条的BGRA图像（透明背景）
        """
        # 创建透明图层
        line_layer = np.zeros((height, width, 4), dtype=np.uint8)
        
        # 计算对角线长度
        diagonal = np.sqrt(width**2 + height**2)
        
        # 生成指定数量的线条
        for _ in range(self.num_lines_per_image):
            # 随机角度
            angle = random.uniform(0, 360)
            angle_rad = np.radians(angle)
            
            # 线条长度（相对于对角线）
            length_factor = random.uniform(*self.line_length_range)
            line_length = diagonal * length_factor
            
            # 随机起点（可能在图像外）
            center_x = random.randint(0, width)
            center_y = random.randint(0, height)
            
            # 计算终点
            dx = line_length / 2 * np.cos(angle_rad)
            dy = line_length / 2 * np.sin(angle_rad)
            
            x1 = int(center_x - dx)
            y1 = int(center_y - dy)
            x2 = int(center_x + dx)
            y2 = int(center_y + dy)
            
            # 线条粗细
            thickness = random.randint(*self.line_thickness_range)
            
            # 线条颜色和透明度
            color, alpha_value = self.generate_line_color()
            
            # 在临时RGB图层上绘制线条
            temp_rgb = np.zeros((height, width, 3), dtype=np.uint8)
            
            if self.line_style == 'solid':
                cv2.line(temp_rgb, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
            elif self.line_style == 'dashed':
                self.draw_dashed_line(temp_rgb, (x1, y1), (x2, y2), color, thickness)
            
            # 创建线条的alpha通道（基于绘制的内容）
            gray = cv2.cvtColor(temp_rgb, cv2.COLOR_BGR2GRAY)
            _, line_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            
            # 应用透明度
            line_mask = (line_mask * alpha_value).astype(np.uint8)
            
            # 将线条合并到图层
            for c in range(3):
                line_layer[:, :, c] = np.where(
                    line_mask > 0,
                    temp_rgb[:, :, c],
                    line_layer[:, :, c]
                )
            
            # 更新alpha通道（取最大值，支持多条线叠加）
            line_layer[:, :, 3] = np.maximum(line_layer[:, :, 3], line_mask)
        
        return line_layer
    
    def draw_dashed_line(self, img, pt1, pt2, color, thickness, dash_length=10):
        """绘制虚线"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        
        if length == 0:
            return
        
        num_dashes = int(length / dash_length)
        
        for i in range(0, num_dashes, 2):
            t1 = i / num_dashes
            t2 = min((i + 1) / num_dashes, 1.0)
            
            px1 = int(x1 + dx * t1)
            py1 = int(y1 + dy * t1)
            px2 = int(x1 + dx * t2)
            py2 = int(y1 + dy * t2)
            
            cv2.line(img, (px1, py1), (px2, py2), color, thickness, cv2.LINE_AA)
    
    def overlay_line_layer(self, img, line_layer):
        """
        将线条图层叠加到物体图像上
        
        参数:
            img: 物体图像 (BGRA)
            line_layer: 线条图层 (BGRA, 透明背景)
        
        返回: 叠加后的图像 (BGRA)
        """
        h, w = img.shape[:2]
        
        # 确保线条图层尺寸匹配
        if line_layer.shape[:2] != (h, w):
            line_layer = cv2.resize(line_layer, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # 提取RGB和alpha通道
        obj_rgb = img[:, :, :3].astype(float)
        obj_alpha = img[:, :, 3].astype(float) / 255.0
        
        line_rgb = line_layer[:, :, :3].astype(float)
        line_alpha = line_layer[:, :, 3].astype(float) / 255.0
        
        # Alpha合成：线条叠加在物体上方
        # 最终alpha = obj_alpha + line_alpha * (1 - obj_alpha)
        final_alpha = np.clip(obj_alpha + line_alpha * (1 - obj_alpha), 0, 1)
        
        # RGB合成：加权混合
        result_rgb = np.zeros_like(obj_rgb)
        
        # 有线条的地方
        mask = line_alpha > 0
        
        for c in range(3):
            # 先放物体，再叠加线条
            result_rgb[:, :, c] = obj_rgb[:, :, c] * obj_alpha
            result_rgb[:, :, c][mask] = (
                result_rgb[:, :, c][mask] * (1 - line_alpha[mask]) +
                line_rgb[:, :, c][mask] * line_alpha[mask]
            )
        
        # 重新归一化（考虑最终alpha）
        valid_mask = final_alpha > 0
        for c in range(3):
            result_rgb[:, :, c][valid_mask] = result_rgb[:, :, c][valid_mask] / final_alpha[valid_mask]
        
        # 组合结果
        result = np.zeros((h, w, 4), dtype=np.uint8)
        result[:, :, :3] = np.clip(result_rgb, 0, 255).astype(np.uint8)
        result[:, :, 3] = (final_alpha * 255).astype(np.uint8)
        
        return result
    
    def add_line_interference(self, img):
        """
        添加线条干扰（使用图层叠加方式）
        """
        h, w = img.shape[:2]
        
        # 1. 生成随机线条图层
        line_layer = self.generate_random_line_layer(w, h)
        
        # 2. 将线条图层叠加到物体上
        result = self.overlay_line_layer(img, line_layer)
        
        return result
    
    def apply_color_variation(self, img):
        """颜色变化（增强饱和度）"""
        rgb = img[:,:,:3].astype(float)
        alpha = img[:,:,3]
        
        # HSV调整
        hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(float)
        
        # 色调偏移 ±15
        hsv[:,:,0] = (hsv[:,:,0] + random.randint(-15, 15)) % 180
        
        # 饱和度增强（配合动画风格）
        hsv[:,:,1] = np.clip(hsv[:,:,1] * random.uniform(1.0, 1.3), 0, 255)
        
        # 亮度 ±20%
        hsv[:,:,2] = np.clip(hsv[:,:,2] * random.uniform(0.85, 1.15), 0, 255)
        
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        result = cv2.cvtColor(rgb, cv2.COLOR_BGR2BGRA)
        result[:,:,3] = alpha
        return result
    
    def apply_brightness_contrast(self, img):
        """亮度对比度调整"""
        rgb = img[:,:,:3].astype(float)
        alpha = img[:,:,3]
        
        # 对比度 0.9-1.2
        contrast = random.uniform(0.9, 1.2)
        rgb = np.clip((rgb - 127.5) * contrast + 127.5, 0, 255)
        
        # 亮度 ±30
        brightness = random.randint(-30, 30)
        rgb = np.clip(rgb + brightness, 0, 255)
        
        result = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2BGRA)
        result[:,:,3] = alpha
        return result
    
    def rotate_image(self, img, angle):
        """旋转图片（保持alpha）"""
        h, w = img.shape[:2]
        center = (w//2, h//2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 计算新尺寸
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        return cv2.warpAffine(img, M, (new_w, new_h), 
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))
    
    def scale_image(self, img, scale):
        """缩放图片"""
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    def add_slight_blur(self, img):
        """轻微模糊（降低概率，保持清晰）"""
        if random.random() < 0.1:  # 从30%降至10%
            ksize = 3  # 只用3x3
            rgb = img[:,:,:3]
            rgb = cv2.GaussianBlur(rgb, (ksize, ksize), 0)
            img[:,:,:3] = rgb
        return img
    
    def generate_variants(self, template_img, obj_id):
        """生成变体"""
        obj_dir = self.output_dir / f'object_{obj_id}'
        obj_dir.mkdir(exist_ok=True)
        
        print(f"\n生成 object_{obj_id} 的 {self.num_variants} 个变体...")
        print(f"  其中 {int(self.num_variants * self.line_interference_ratio)} 个带线条干扰")
        
        # 保存原始版本
        base_img = template_img.copy()
        cv2.imwrite(str(obj_dir / 'object_000.png'), base_img)
        
        variant_count = 1
        line_count = 0
        
        # 生成变体
        for i in range(self.num_variants - 1):
            img = template_img.copy()
            
            # 1. 旋转（80%概率）
            if random.random() < 0.8:
                angle = random.uniform(-90, 90)
                img = self.rotate_image(img, angle)
            
            # 2. 缩放（60%概率）
            if random.random() < 0.6:
                scale = random.uniform(0.7, 1.4)
                img = self.scale_image(img, scale)
            
            # 3. 颜色变化（70%概率）
            if random.random() < 0.7:
                img = self.apply_color_variation(img)
            
            # 4. 亮度对比度（80%概率）
            if random.random() < 0.8:
                img = self.apply_brightness_contrast(img)
            
            # 5. 轻微模糊（10%概率）
            img = self.add_slight_blur(img)
            
            # 6. 裁剪到内容
            img = self.crop_to_content(img)
            
            # 7. 添加线条干扰（按比例）- 使用图层叠加方式
            has_line = random.random() < self.line_interference_ratio
            suffix = ''
            
            if has_line:
                img = self.add_line_interference(img)
                suffix = '_line'
                line_count += 1
            
            # 8. 保存
            output_path = obj_dir / f'object_{variant_count:03d}{suffix}.png'
            cv2.imwrite(str(output_path), img)
            variant_count += 1
            
            if (i + 1) % 30 == 0:
                print(f"  已生成 {i + 1}/{self.num_variants - 1}")
        
        print(f"✓ object_{obj_id} 完成: {variant_count} 个变体 ({line_count} 个带线条)")
        
        # 创建预览
        self.create_preview(obj_dir, obj_id)
    
    def create_preview(self, obj_dir, obj_id):
        """创建预览图"""
        images = sorted(obj_dir.glob('object_*.png'))[:16]
        
        if len(images) == 0:
            return
        
        # 读取图片
        imgs = []
        for img_path in images:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                # 调整大小
                h, w = img.shape[:2]
                scale = min(150/w, 150/h)
                new_w, new_h = int(w*scale), int(h*scale)
                img = cv2.resize(img, (new_w, new_h))
                
                # 转换为有背景的图片
                if img.shape[2] == 4:
                    bg = np.ones((150, 150, 3), dtype=np.uint8) * 240
                    alpha = img[:,:,3:] / 255.0
                    rgb = img[:,:,:3]
                    
                    y_offset = (150 - new_h) // 2
                    x_offset = (150 - new_w) // 2
                    
                    for c in range(3):
                        bg[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c] = \
                            bg[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c] * (1 - alpha[:,:,0]) + \
                            rgb[:,:,c] * alpha[:,:,0]
                    
                    imgs.append(bg)
        
        # 创建网格
        rows = []
        for i in range(0, len(imgs), 4):
            row = np.hstack(imgs[i:i+4])
            rows.append(row)
        
        if rows:
            preview = np.vstack(rows)
            cv2.imwrite(str(obj_dir / '_preview.png'), preview)
    
    def process_all(self):
        """处理所有模板"""
        templates = sorted(self.template_dir.glob('t*.png'))
        
        if len(templates) == 0:
            print(f"❌ 没有找到模板文件在 {self.template_dir}")
            return
        
        print("=" * 70)
        print("从手动模板生成增强物体（图层叠加式线条干扰）")
        print("=" * 70)
        print(f"模板目录: {self.template_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"找到 {len(templates)} 个模板")
        
        for i, template_path in enumerate(templates, 1):
            print(f"\n处理模板 {i}/{len(templates)}: {template_path.name}")
            
            # 加载模板
            img = self.load_template(template_path)
            print(f"  尺寸: {img.shape[1]}×{img.shape[0]}")
            
            # 轻微增强边缘
            img = self.enhance_edges(img)
            
            # 裁剪
            img = self.crop_to_content(img)
            print(f"  裁剪后: {img.shape[1]}×{img.shape[0]}")
            
            # 生成变体
            self.generate_variants(img, i)
        
        print("\n" + "=" * 70)
        print("✓ 所有物体生成完成！")
        print("=" * 70)
        print(f"\n可以查看预览: ls -lh {self.output_dir}/object_*/_preview.png")

if __name__ == '__main__':
    preparer = ObjectPreparer()
    
    # ============ 可以在这里调整参数 ============
    # preparer.num_variants = 200                  # 改为200个变体
    # preparer.line_interference_ratio = 0.5       # 改为50%带线条
    # preparer.num_lines_per_image = 2             # 改为每图2条线
    # preparer.line_style = 'dashed'               # 改为虚线
    # preparer.line_thickness_range = (1, 2)       # 改为更粗的线条
    
    preparer.process_all()
