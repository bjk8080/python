import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt

# ----------------------------
# 샘플 이미지 생성 함수
# ----------------------------

def create_high_quality_sample():
    image = np.ones((100, 400, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'perfect OCR Text', (20, 60), font, 1.2, (0, 0, 0), 2)
    cv2.imwrite('high_quality_sample.jpg', image)
    return image

def create_medium_quality_sample():
    image = np.ones((100, 400, 3), dtype=np.uint8) * 255
    noise = np.random.normal(0, 15, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'Noisy OCR Text', (20, 60), font, 1.2, (0, 0, 0), 2)
    cv2.imwrite('medium_quality_sample.jpg', image)
    return image

def create_low_quality_sample():
    image = np.ones((100, 400, 3), dtype=np.uint8) * 255
    noise = np.random.normal(0, 30, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 텍스트 색상을 진하게 변경
    cv2.putText(image, 'Blurry OCR Text', (20, 60), font, 1.2, (0, 0, 0), 2)

    # 회전과 블러 적용
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 5, 1)
    image = cv2.warpAffine(image, M, (cols, rows))
    image = cv2.GaussianBlur(image, (3,3), 0)

    cv2.imwrite('low_quality_sample.jpg', image)
    return image

# ----------------------------
# OCR 수행 및 품질 평가
# ----------------------------

def ocr_quality_demo():
    samples = {
        'high_quality': create_high_quality_sample(),
        'medium_quality': create_medium_quality_sample(),
        'low_quality': create_low_quality_sample()
    }

    results = {}

    for quality, image in samples.items():
        text = pytesseract.image_to_string(image, lang='eng')
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        # int 또는 float 형태 confidence 안전 계산
        confidences = [float(conf) for conf in data['conf'] if conf > 0]
        avg_confidence = np.mean(confidences) if confidences else 0

        results[quality] = {
            'text': text.strip(),
            'confidence': avg_confidence,
            'image': image
        }

        print(f"{quality.upper()}:")
        print(f" 텍스트: '{text.strip()}'")
        print(f" 평균 신뢰도: {avg_confidence:.1f}%\n")

    return results

# ----------------------------
# 실행 및 시각화
# ----------------------------

quality_results = ocr_quality_demo()

plt.figure(figsize=(15,10))
plt.rc('font', family='malgun Gothic')

# 기본 OCR 이미지
plt.subplot(2,3,1)
plt.imshow(cv2.cvtColor(quality_results['high_quality']['image'], cv2.COLOR_BGR2RGB))
plt.title('기본 OCR 예제')
plt.axis('off')

# 나머지 샘플 이미지
for i, (quality, result) in enumerate(quality_results.items(), 2):
    plt.subplot(2,3,i)
    plt.imshow(cv2.cvtColor(result['image'], cv2.COLOR_BGR2RGB))
    plt.title(f'{quality.title()}\n신뢰도: {result["confidence"]:.1f}%')
    plt.axis('off')

plt.tight_layout()
plt.show()
