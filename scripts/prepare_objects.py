import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import shutil
from scipy.ndimage import map_coordinates, gaussian_filter

class ObjectPreparer:
    """物体准备器 - 增强版"""
    
    def __init__(self):
        self.raw_dir = Path('data/raw_templates')
        self.output_dir = Path('data/objects')
        
    def remove_background_advanced(self, img):
        """高级背景去除（使用GrabCut）"""
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # 创建 mask
        mask = np.zeros(img.shape[:2], np.uint8)
        
        # 定义矩形区域（假设物体在中心）
        h, w = img.shape[:2]
        margin = int(min(h, w) * 0.05)  # 5% 边距
        rect = (margin, margin, w - 2*margin, h - 2*margin)
        
        # GrabCut 背景/前景分离
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        except:
            # 如果 GrabCut 失败，使用传统方法
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask2 = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 形态学操作清理
        kernel = np.ones((5, 5), np.uint8)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # 找到最大连通区域
        contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            mask_clean = np.zeros_like(mask2)
            cv2.drawContours(mask_clean, [largest_contour], -1, 1, -1)
            mask2 = mask_clean
        
        # 边缘羽化（让边缘更自然）
        mask2 = (mask2 * 255).astype(np.uint8)
        mask2 = cv2.GaussianBlur(mask2, (7, 7), 2)
        
        # 创建 RGBA 图像
        result = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
        result[:,:,:3] = img
        result[:,:,3] = mask2
        
        return result
    
    def crop_object(self, img, margin=20):
        """裁剪到物体边界（增加边距）"""
        if img.shape[2] == 4:
            mask = img[:,:,3]
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        coords = cv2.findNonZero(mask)
        if coords is None:
            return img
        
        x, y, w, h = cv2.boundingRect(coords)
        
        # 增加边距
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img.shape[1] - x, w + 2*margin)
        h = min(img.shape[0] - y, h + 2*margin)
        
        return img[y:y+h, x:x+w]
    
    def elastic_transform(self, image, alpha=50, sigma=5):
        """弹性变形（模拟物体形变）"""
        if image.shape[2] != 4:
            return image
            
        shape = image.shape[:2]
        
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = (np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)))
        
        result = np.zeros_like(image)
        for i in range(4):  # 处理 RGBA 4 个通道
            channel = map_coordinates(image[:,:,i], indices, order=1, mode='reflect')
            result[:,:,i] = channel.reshape(shape)
        
        return result
    
    def add_shadow(self, img):
        """添加阴影效果"""
        if img.shape[2] != 4:
            return img
        
        result = img.copy()
        
        # 随机阴影方向和强度
        angle = np.random.uniform(0, 360)
        strength = np.random.uniform(0.3, 0.7)
        
        # 创建渐变阴影
        h, w = img.shape[:2]
        shadow = np.ones((h, w), dtype=np.float32)
        
        # 根据角度创建渐变
        angle_rad = np.radians(angle)
        for i in range(h):
            for j in range(w):
                dist = (i * np.sin(angle_rad) + j * np.cos(angle_rad)) / max(h, w)
                shadow[i, j] = 1 - strength * max(0, dist)
        
        # 应用阴影到 RGB 通道
        for c in range(3):
            result[:,:,c] = (result[:,:,c] * shadow).astype(np.uint8)
        
        return result
    
    def add_noise(self, img):
        """添加噪声"""
        if img.shape[2] != 4:
            return img
        
        result = img.copy()
        noise_type = np.random.choice(['gaussian', 'salt_pepper', 'none'], p=[0.3, 0.2, 0.5])
        
        if noise_type == 'gaussian':
            # 高斯噪声
            sigma = np.random.uniform(1, 5)
            gauss = np.random.normal(0, sigma, (img.shape[0], img.shape[1], 3))
            result[:,:,:3] = np.clip(result[:,:,:3] + gauss, 0, 255).astype(np.uint8)
            
        elif noise_type == 'salt_pepper':
            # 椒盐噪声
            prob = np.random.uniform(0.001, 0.01)
            mask = np.random.random((img.shape[0], img.shape[1]))
            result[:,:,:3][mask < prob/2] = 0
            result[:,:,:3][mask > 1 - prob/2] = 255
        
        return result
    
    def add_blur(self, img):
        """添加模糊效果"""
        blur_type = np.random.choice(['none', 'gaussian', 'motion'], p=[0.5, 0.3, 0.2])
        
        if blur_type == 'gaussian':
            # 高斯模糊
            kernel_size = np.random.choice([3, 5, 7])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            
        elif blur_type == 'motion':
            # 运动模糊
            size = np.random.randint(5, 15)
            kernel = np.zeros((size, size))
            kernel[int((size-1)/2), :] = np.ones(size)
            kernel = kernel / size
            
            # 随机旋转模糊核
            angle = np.random.uniform(0, 360)
            M = cv2.getRotationMatrix2D((size/2, size/2), angle, 1)
            kernel = cv2.warpAffine(kernel, M, (size, size))
            
            # 分别处理 RGBA
            result = img.copy()
            for c in range(min(3, img.shape[2])):  # 只模糊 RGB
                result[:,:,c] = cv2.filter2D(img[:,:,c], -1, kernel)
            img = result
        
        return img
    
    def perspective_transform(self, img):
        """透视变换（模拟不同角度观察）"""
        h, w = img.shape[:2]
        
        # 随机透视强度
        strength = np.random.uniform(0.1, 0.3)
        
        # 源点
        src_points = np.float32([
            [0, 0],
            [w-1, 0],
            [w-1, h-1],
            [0, h-1]
        ])
        
        # 目标点（添加随机扰动）
        dst_points = np.float32([
            [np.random.uniform(0, w*strength), np.random.uniform(0, h*strength)],
            [w-1-np.random.uniform(0, w*strength), np.random.uniform(0, h*strength)],
            [w-1-np.random.uniform(0, w*strength), h-1-np.random.uniform(0, h*strength)],
            [np.random.uniform(0, w*strength), h-1-np.random.uniform(0, h*strength)]
        ])
        
        # 透视变换矩阵
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # 应用变换
        result = cv2.warpPerspective(img, M, (w, h), 
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0,0,0,0))
        
        return result
    
    def generate_variants(self, img, num_variants=50):
        """生成增强的变体"""
        variants = [img]  # 原始图像
        
        for i in range(num_variants - 1):
            variant = img.copy()
            
            # 转换为 PIL 进行基础增强
            if variant.shape[2] == 4:
                pil_img = Image.fromarray(cv2.cvtColor(variant, cv2.COLOR_BGRA2RGBA))
            else:
                pil_img = Image.fromarray(cv2.cvtColor(variant, cv2.COLOR_BGR2RGB))
            
            # === 1. 旋转（更大范围：-180° 到 +180°）===
            if np.random.random() > 0.2:
                angle = np.random.uniform(-180, 180)
                pil_img = pil_img.rotate(angle, expand=True, fillcolor=(0,0,0,0))
            
            # === 2. 缩放（更大范围：0.5x - 2.0x）===
            if np.random.random() > 0.2:
                scale = np.random.uniform(0.5, 2.0)
                new_size = (int(pil_img.width * scale), int(pil_img.height * scale))
                if new_size[0] > 10 and new_size[1] > 10:  # 确保不会太小
                    pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
            
            # === 3. 颜色调整（更大范围）===
            if np.random.random() > 0.3:
                # 亮度
                enhancer = ImageEnhance.Brightness(pil_img)
                pil_img = enhancer.enhance(np.random.uniform(0.6, 1.4))
                
                # 对比度
                enhancer = ImageEnhance.Contrast(pil_img)
                pil_img = enhancer.enhance(np.random.uniform(0.7, 1.3))
                
                # 饱和度
                enhancer = ImageEnhance.Color(pil_img)
                pil_img = enhancer.enhance(np.random.uniform(0.7, 1.3))
                
                # 锐度
                enhancer = ImageEnhance.Sharpness(pil_img)
                pil_img = enhancer.enhance(np.random.uniform(0.5, 2.0))
            
            # === 4. 翻转 ===
            if np.random.random() > 0.5:
                pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.random() > 0.8:  # 垂直翻转概率较低
                pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
            
            # 转回 OpenCV 格式
            variant = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)
            
            # === 5. 透视变换（30% 概率）===
            if np.random.random() > 0.7 and variant.shape[0] > 20 and variant.shape[1] > 20:
                variant = self.perspective_transform(variant)
            
            # === 6. 弹性变形（20% 概率）===
            if np.random.random() > 0.8 and variant.shape[0] > 20 and variant.shape[1] > 20:
                variant = self.elastic_transform(variant, 
                                                 alpha=np.random.uniform(20, 50),
                                                 sigma=np.random.uniform(4, 8))
            
            # === 7. 添加阴影（30% 概率）===
            if np.random.random() > 0.7:
                variant = self.add_shadow(variant)
            
            # === 8. 添加模糊（30% 概率）===
            if np.random.random() > 0.7:
                variant = self.add_blur(variant)
            
            # === 9. 添加噪声（40% 概率）===
            if np.random.random() > 0.6:
                variant = self.add_noise(variant)
            
            variants.append(variant)
        
        return variants
    
    def process_all(self):
        """处理所有模板"""
        print("=" * 70)
        print("🎨 准备目标物体数据（增强版）")
        print("=" * 70)
        print(f"📁 输入目录: {self.raw_dir}")
        print(f"📁 输出目录: {self.output_dir}")
        print(f"")
        print("🔧 增强功能:")
        print("  ✓ 高级背景去除（GrabCut）")
        print("  ✓ 边缘羽化处理")
        print("  ✓ 大范围旋转（-180° 到 +180°）")
        print("  ✓ 大范围缩放（0.5x - 2.0x）")
        print("  ✓ 透视变换（模拟不同角度）")
        print("  ✓ 弹性变形")
        print("  ✓ 阴影效果")
        print("  ✓ 模糊效果（高斯 + 运动）")
        print("  ✓ 噪声（高斯 + 椒盐）")
        print("  ✓ 颜色增强（亮度/对比度/饱和度/锐度）")
        print("  ✓ 50 个变体/物体")
        print("")
        
        # 查找所有模板
        templates = sorted(self.raw_dir.glob('t*.png'))
        
        if len(templates) == 0:
            print(f"❌ 错误：在 {self.raw_dir} 中没有找到模板文件（t1.png, t2.png, ...）")
            return
        
        print(f"🔍 找到 {len(templates)} 个模板\n")
        print("=" * 70)
        
        # 处理每个模板
        for i, template_path in enumerate(templates, 1):
            print(f"\n[{i}/{len(templates)}] 📷 处理 {template_path.name}")
            print("-" * 70)
            
            # 创建输出目录
            obj_dir = self.output_dir / f'object_{i}'
            obj_dir.mkdir(parents=True, exist_ok=True)
            
            # 读取模板
            img = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
            
            if img is None:
                print(f"  ❌ 无法读取图像")
                continue
            
            print(f"  📐 原始尺寸: {img.shape[1]}×{img.shape[0]}")
            
            # 如果没有 alpha 通道，去除背景
            if img.shape[2] != 4:
                print(f"  🎭 去除背景（GrabCut）...")
                img = self.remove_background_advanced(img)
            else:
                # 即使有 alpha 通道，也进行边缘羽化
                print(f"  🎨 边缘羽化处理...")
                mask = img[:,:,3]
                mask = cv2.GaussianBlur(mask, (7, 7), 2)
                img[:,:,3] = mask
            
            # 裁剪到物体边界
            img = self.crop_object(img, margin=20)
            print(f"  ✂️  裁剪后尺寸: {img.shape[1]}×{img.shape[0]}")
            
            # 保存原始版本
            cv2.imwrite(str(obj_dir / f'object_{i}_000.png'), img)
            
            # 生成变体
            print(f"  🔄 生成 50 个增强变体...")
            variants = self.generate_variants(img, num_variants=50)
            
            # 保存变体（带进度）
            for j, variant in enumerate(variants[1:], 1):
                variant = self.crop_object(variant, margin=20)
                cv2.imwrite(str(obj_dir / f'object_{i}_{j:03d}.png'), variant)
                
                if j % 10 == 0:
                    print(f"    进度: {j}/50")
            
            print(f"  ✅ 完成：生成 {len(variants)} 个变体")
        
        print("\n" + "=" * 70)
        print("✅ 物体准备完成！")
        print("=" * 70)
        print(f"\n📊 目录结构：")
        
        total_images = 0
        for i in range(1, len(templates) + 1):
            obj_dir = self.output_dir / f'object_{i}'
            if obj_dir.exists():
                count = len(list(obj_dir.glob('*.png')))
                total_images += count
                print(f"  📁 data/objects/object_{i}/  ({count} 张)")
        
        print(f"\n🎉 总计: {total_images} 张物体图像")
        print("")
        print("=" * 70)
        print("🚀 下一步:")
        print("  运行: python scripts/3_synthesize_training_data_v2_fixed.py")
        print("=" * 70)

def main():
    preparer = ObjectPreparer()
    preparer.process_all()

if __name__ == '__main__':
    main()
