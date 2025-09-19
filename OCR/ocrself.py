import cv2
import pytesseract
from matplotlib import pyplot as plt

image_path = "your_photo.jpg" 
image = cv2.imread(image_path)

if image is None:
    print("이미지를 찾을 수 없습니다.")
    exit()

self = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

self = cv2.medianBlur(self, 3)

_, image = cv2.threshold(self, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

text = pytesseract.image_to_string(image, lang='eng')

print("===== 인식된 텍스트 =====")
print(text)
