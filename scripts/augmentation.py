import albumentations as A
import cv2
import numpy as np
from PIL import Image

def get_train_transforms():
    """
    재활용품 VQA 학습용 데이터 증강 파이프라인
    - 주의: 텍스트(라벨)를 읽어야 하므로 좌우 반전(HorizontalFlip)은 제외했습니다.
    """
    return A.Compose([
        # 1. 기하학적 변형 (약간의 회전 및 크기 조절로 다양한 촬영 구도 모사)
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, 
                           p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
        
        # 2. 색상 및 조명 변형 (실내 형광등, 야외 햇빛, 그림자 등 다양한 조명 모사)
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        
        # 3. 노이즈 및 화질 저하 (저화질 스마트폰 카메라 모사)
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.ImageCompression(quality_lower=80, quality_upper=100, p=0.2),
    ])

def get_valid_transforms():
    """검증/테스트용 파이프라인 (증강 없이 원본 유지)"""
    return A.Compose([])

def apply_augmentation(image: Image.Image, transform) -> Image.Image:
    """
    PIL Image를 받아 Albumentations 증강을 적용한 후 다시 PIL Image로 반환
    """
    if transform is None:
        return image
        
    # PIL -> Numpy (OpenCV format)
    image_np = np.array(image.convert("RGB"))
    
    # Augmentation 적용
    augmented = transform(image=image_np)
    
    # Numpy -> PIL
    return Image.fromarray(augmented['image'])