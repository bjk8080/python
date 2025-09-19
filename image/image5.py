# pip install ultralytics
# 욜로는 원본이미지를 이미지 세그먼테이션을 이용한 뒤에 그 사진이 무엇인지 바운딩 박스로 체크하는 프로그램
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

class YOLOSegmentation:
    """YOLO를 활용한 인스턴스 세그먼테이션"""
    # YOLOSegmentation 클래스: YOLO(You Only Look Once) 모델을 사용하여 인스턴스 세그먼테이션을 수행하는 클래스
    # 인스턴스 세그먼테이션: 객체 검출과 동시에 각 객체의 정확한 경계(마스크)를 픽셀 단위로 분할하는 기술
    # YOLO 사용 이유: 실시간 처리가 가능하고 정확도가 높으며, 사용이 간편함
    
    def __init__(self, model_name='yolov8n-seg.pt'):
        # __init__ 메서드: 클래스 인스턴스 초기화 시 호출되는 생성자 함수
        # 매개변수 설명:
        # - model_name (str): 사용할 YOLO 세그먼테이션 모델의 이름/경로
        #   * 기본값 'yolov8n-seg.pt': YOLOv8 nano 세그먼테이션 모델 (가장 빠르고 가벼움)
        #   * 다른 옵션: yolov8s-seg.pt (small), yolov8m-seg.pt (medium), 
        #               yolov8l-seg.pt (large), yolov8x-seg.pt (extra large)
        #   * 모델 크기가 클수록 정확도는 높아지지만 속도는 느려짐
        
        # YOLOv8 세그먼테이션 모델 로드
        self.model = YOLO(model_name)
        # YOLO 객체 생성 이유: Ultralytics에서 제공하는 사전 훈련된 모델을 쉽게 사용하기 위함
        # 모델이 없으면 자동으로 다운로드됨
    
    def predict_and_visualize(self, image_path, confidence=0.5):
        """이미지에서 세그먼테이션 수행 및 시각화"""
        # predict_and_visualize 메서드: 입력 이미지에 대해 세그먼테이션을 수행하고 결과를 시각화하는 함수
        # 매개변수 설명:
        # - image_path (str): 분석할 이미지 파일의 경로
        #   * 지원 형식: jpg, png, bmp 등 일반적인 이미지 형식
        # - confidence (float): 객체 검출 신뢰도 임계값 (0.0 ~ 1.0)
        #   * 기본값 0.5: 50% 이상 확신할 때만 객체로 인식
        #   * 값이 높을수록 더 확실한 객체만 검출 (정밀도 향상, 재현율 감소)
        #   * 값이 낮을수록 더 많은 객체 검출 (재현율 향상, 정밀도 감소)
        
        # 예측 수행
        results = self.model(image_path, conf=confidence)
        # model() 호출: 이미지에 대해 추론을 수행
        # conf 매개변수: confidence threshold 설정으로 해당 값 이상의 확신을 가진 객체만 검출
        
        # 결과 시각화
        for r in results:
            # results는 리스트 형태로 반환되므로 반복문으로 처리
            # 일반적으로 단일 이미지의 경우 하나의 결과만 포함
            
            # 원본 이미지
            img = cv2.imread(image_path)
            # cv2.imread(): OpenCV를 사용하여 이미지를 BGR 형식으로 읽기
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # cv2.cvtColor(): BGR을 RGB로 변환 (matplotlib는 RGB 형식을 사용하므로)
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            # plt.subplots(): 1행 3열의 서브플롯 생성
            # figsize=(18, 6): 전체 그림 크기를 18x6 인치로 설정
            # 3개 서브플롯 이유: 원본, 검출 결과, 세그먼테이션 마스크를 각각 표시하기 위함
            
            # 원본 이미지
            axes[0].imshow(img_rgb)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            # imshow(): 이미지 표시
            # set_title(): 서브플롯 제목 설정
            # axis('off'): 축 정보 숨기기 (이미지만 깔끔하게 표시)
            
            # 바운딩 박스와 라벨
            img_with_boxes = r.plot()
            # r.plot(): YOLO 결과에 바운딩 박스와 라벨을 그린 이미지 반환
            # 자동으로 클래스명, 신뢰도, 바운딩 박스를 시각화
            axes[1].imshow(img_with_boxes)
            axes[1].set_title('Detection Results')
            axes[1].axis('off')
            
            # 마스크만 표시
            if r.masks is not None:
                # r.masks: 세그먼테이션 마스크 정보 (있을 경우에만 처리)
                # None 체크 이유: 객체가 검출되지 않거나 세그먼테이션이 실패한 경우 대비
                masks = r.masks.data.cpu().numpy()
                # .data.cpu().numpy(): 텐서 데이터를 CPU로 이동 후 numpy 배열로 변환
                # GPU에서 처리된 결과를 CPU 메모리로 가져와서 일반적인 배열 연산 수행
                combined_mask = np.zeros_like(img_rgb)
                # np.zeros_like(): 원본 이미지와 같은 크기의 빈 배열 생성
                # 모든 마스크를 합성할 캔버스 역할
                
                for i, mask in enumerate(masks):
                    # enumerate(): 인덱스와 값을 함께 반환하는 반복문
                    # 각 검출된 객체별로 개별 마스크 처리
                    
                    # 마스크 크기를 원본 이미지에 맞게 조정
                    mask_resized = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]))
                    # cv2.resize(): 마스크 크기를 이미지 크기에 맞게 조정
                    # (img_rgb.shape[1], img_rgb.shape[0]): (너비, 높이) 순서로 크기 지정
                    # 크기 조정 이유: YOLO 모델 출력 해상도와 원본 이미지 해상도가 다를 수 있음
                    
                    # 컬러 마스크 생성
                    color = np.random.randint(0, 255, 3)
                    # np.random.randint(): 0~255 사이의 랜덤한 RGB 색상 생성
                    # 각 객체를 다른 색상으로 구분하기 위함
                    colored_mask = np.zeros_like(img_rgb)
                    colored_mask[mask_resized > 0.5] = color
                    # mask_resized > 0.5: 마스크 값이 0.5 이상인 픽셀을 객체로 판단
                    # 해당 픽셀에 생성된 색상 적용
                    
                    combined_mask = cv2.addWeighted(combined_mask, 1, colored_mask, 0.7, 0)
                    # cv2.addWeighted(): 두 이미지를 가중합으로 합성
                    # (combined_mask, 1, colored_mask, 0.7, 0): 기존 마스크 100% + 새 마스크 70%
                    # 여러 객체의 마스크를 겹쳐서 표시
                
                # 마스크를 원본 이미지에 오버레이
                result_img = cv2.addWeighted(img_rgb, 0.6, combined_mask, 0.4, 0)
                # 원본 이미지 60% + 마스크 40%로 합성하여 반투명 오버레이 효과
                # 이유: 원본 이미지도 보이면서 세그먼테이션 결과도 명확하게 표시
                axes[2].imshow(result_img)
                axes[2].set_title('Segmentation Masks')
            else:
                # 마스크가 없는 경우 대체 텍스트 표시
                axes[2].text(0.5, 0.5, 'No masks detected', 
                           transform=axes[2].transAxes, ha='center', va='center')
                # transform=axes[2].transAxes: 축 좌표계 사용 (0~1 범위)
                # ha='center', va='center': 수평, 수직 중앙 정렬
                axes[2].set_title('No Segmentation Results')
            
            axes[2].axis('off')
            plt.tight_layout()
            # plt.tight_layout(): 서브플롯 간 간격 자동 조정으로 겹침 방지
            plt.show()
            # plt.show(): 그래프 화면에 표시
            
            # 검출된 객체 정보 출력
            if r.boxes is not None:
                # r.boxes: 바운딩 박스 정보 (객체가 검출된 경우에만 존재)
                print(f"검출된 객체 수: {len(r.boxes)}")
                # len(r.boxes): 검출된 객체의 총 개수
                for i, box in enumerate(r.boxes):
                    # 각 검출된 객체에 대한 상세 정보 출력
                    class_id = int(box.cls[0])
                    # box.cls[0]: 예측된 클래스 ID (정수형으로 변환)
                    # 0: 첫 번째 클래스 ID
                    # 첫번째 클래스 아이디 가져오는 이유 : 예측된 클래스 ID가 여러개일 수 있기 때문에, 첫번째 클래스 아이디를 가져옴
                    confidence = float(box.conf[0])
                    # box.conf[0]: 해당 객체에 대한 신뢰도 (실수형으로 변환)
                    class_name = self.model.names[class_id]
                    # self.model.names: 클래스 ID를 실제 클래스명으로 매핑하는 딕셔너리
                    print(f"객체 {i+1}: {class_name} (신뢰도: {confidence:.2f})")
                    # {confidence:.2f}: 신뢰도를 소수점 둘째 자리까지 표시

# 사용 예제
def demo_yolo_segmentation():
    """YOLO 세그먼테이션 데모"""
    # demo_yolo_segmentation 함수: YOLO 세그먼테이션 기능을 테스트하고 시연하는 함수
    # 매개변수 없음: 단순 데모 목적으로 모델 초기화와 기본 정보만 출력
    # 함수 목적: 사용자가 코드를 실행했을 때 모델이 정상적으로 로드되는지 확인
    
    yolo_seg = YOLOSegmentation()
    # YOLOSegmentation 인스턴스 생성
    # 기본 모델(yolov8n-seg.pt) 사용하여 가장 빠른 테스트 환경 제공
    
    # 실제 이미지 경로로 테스트
    yolo_seg.predict_and_visualize('image.jpg', confidence=0.5)
    
    print("YOLO 세그먼테이션 모델이 준비되었습니다.")
    # 모델 로딩 완료 메시지
    print(f"지원하는 클래스: {list(yolo_seg.model.names.values())}")
    # 모델이 인식할 수 있는 모든 객체 클래스 목록 출력
    # COCO 데이터셋 기준 80개 클래스 (사람, 자동차, 동물 등)
    # 사용자가 어떤 객체들을 검출할 수 있는지 미리 확인 가능

demo_yolo_segmentation()
# 스크립트 실행 시 데모 함수 자동 호출
# 모듈 import 시에는 실행되지 않고, 직접 스크립트 실행 시에만 데모 실행