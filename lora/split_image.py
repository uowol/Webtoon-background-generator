import numpy as np
import shutil
import os
from PIL import Image

def divide_image_into_patches(image):
    """
    이미지를 최소 256 픽셀 크기의 정사각형 패치로 나누고 모든 패치를 반환합니다.
    
    매개변수:
    image: HxWxC 크기의 이미지 데이터 (numpy array)
    
    반환값:
    patches: 모든 패치를 포함하는 리스트
    """
    H, W, C = image.shape # 이미지의 높이, 너비, 채널 수를 얻습니다.
    
    patch_size = 200
    
    # 패치 리스트 초기화
    patches = []
    
    # 이미지를 패치로 나눕니다.
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            # 패치 추출: 범위를 넘어가지 않도록 min 함수를 사용합니다.
            if i+patch_size > H: continue
            if j+patch_size > W: continue
            patch = image[i:min(i+patch_size, H), j:min(j+patch_size, W), :]
            # print(patch.sum(), patch[:,:,3].sum(), patch.shape, patch.min(), patch.max())
            # if patch[:,:,3].sum() < 22950000: continue 
            if patch.sum() > 70000000: continue
            patches.append(patch)
    
    return patches


def pil_to_numpy(image):
    """
    PIL 이미지를 NumPy 배열로 변환합니다.
    
    매개변수:
    image: PIL 이미지 객체
    
    반환값:
    numpy_array: NumPy 배열
    """
    numpy_array = np.array(image)
    return numpy_array

def numpy_to_pil(numpy_array):
    """
    NumPy 배열을 PIL 이미지 형식으로 변환합니다.
    
    매개변수:
    numpy_array: NumPy 배열
    
    반환값:
    image: PIL 이미지 객체
    """
    image = Image.fromarray(numpy_array)
    return image


# pil_image = Image.open("/data/ephemeral/home/lora-0.1.7/data/temp/마음소리.png")
# try:
#     shutil.rmtree("/data/ephemeral/home/lora-0.1.7/data/temp2")
# except:
#     pass
# os.makedirs("/data/ephemeral/home/lora-0.1.7/data/temp2")
# np_image = pil_to_numpy(pil_image)
# np_patches = divide_image_into_patches(np_image)
# for i, np_patch in enumerate(np_patches):
#     pil_patch = numpy_to_pil(np_patch)
#     pil_patch.save(f"/data/ephemeral/home/lora-0.1.7/data/temp2/temp_{i}.png")