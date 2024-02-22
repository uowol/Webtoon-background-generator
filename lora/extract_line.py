import cv2
import numpy as np

# opencv-python==4.8.1.78
# opencv-python-headless==4.9.0.80


def extract_lines(image_path, output_path):
    # 이미지를 읽어옵니다.
    original_img = cv2.imread(image_path)
    tone_image = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)  # 그레이스케일로 변환
    line_art = cv2.Canny(tone_image, 50, 200)  # Canny 엣지 검출기 사용
    
    # 선화 이미지와 명암 이미지를 3채널로 변환
    line_art_3ch = cv2.cvtColor(line_art, cv2.COLOR_GRAY2BGR)
    tone_image_3ch = cv2.cvtColor(tone_image, cv2.COLOR_GRAY2BGR)
    
    # 원본 이미지에서 선화 이미지와 명암 이미지를 빼서 색상 이미지 생성
    # color_img = cv2.subtract(original_img, line_art_3ch)
    # color_img = cv2.subtract(color_img, tone_image_3ch)
    color_img = cv2.subtract(original_img, tone_image_3ch)
    
    cv2.imwrite('/data/ephemeral/home/lora-0.1.7/data/temp/line_art.jpg', line_art)
    cv2.imwrite('/data/ephemeral/home/lora-0.1.7/data/temp/tone_image.jpg', tone_image)
    cv2.imwrite('/data/ephemeral/home/lora-0.1.7/data/temp/color_image.jpg', color_img)

# 이미지 파일 경로를 지정하고 함수를 호출합니다.
image_path = "/data/ephemeral/home/lora-0.1.7/data/temp/Screenshot 2024-02-19 at 14.48.36.JPG"
output_path = "/data/ephemeral/home/lora-0.1.7/data/temp/test_line.jpg"
extract_lines(image_path, output_path)
