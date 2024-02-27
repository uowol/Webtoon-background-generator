import cv2
import numpy as np

def redraw_image_with_emphasized_outlines(image_path):
    # 이미지를 읽어옵니다.
    img = cv2.imread(image_path)
    
    # 그레이스케일로 변환합니다.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Gaussian Blur를 적용하여 이미지를 부드럽게 합니다.
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Canny 엣지 검출기를 사용하여 윤곽선을 추출합니다.
    edges = cv2.Canny(blurred, threshold1=100, threshold2=200)
    
    # 윤곽선 이미지를 3채널로 변환합니다.
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # 원본 이미지와 윤곽선 이미지를 합성합니다.
    result = cv2.addWeighted(img, 0.8, edges_colored, 0.2, 0)
    
    # 결과 이미지를 표시합니다.
    cv2.imshow('Redrawn Image with Emphasized Outlines', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 결과 이미지를 파일로 저장합니다.
    cv2.imwrite('redrawn_image.jpg', result)

# 이미지 파일 경로를 지정하고 함수를 호출합니다.
image_path = 'your_background_image_path_here.jpg'
redraw_image_with_emphasized_outlines(image_path)
