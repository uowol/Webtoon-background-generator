import cv2
import numpy as np

def simplify_image(image_path, K=8):
    # 이미지를 읽고 RGB로 변환
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 이미지를 2D로 변환 (색상 클러스터링을 위해)
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    
    # K-평균 클러스터링 실행
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # 각 픽셀에 대한 클러스터 중심 색상 할당
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    
    # 결과 이미지를 RGB에서 BGR로 변환 후 저장
    result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('/data/ephemeral/home/lora-0.1.7/data/temp/simplified_image.jpg', result_image_bgr)

    print("이미지 단순화가 완료되었습니다.")

# 이미지 파일 경로를 지정하고 함수를 호출합니다.
image_path = '/data/ephemeral/home/lora-0.1.7/data/temp/rand1.jpeg'
simplify_image(image_path, K=8)  # K는 사용할 클러스터(색상)의 수
