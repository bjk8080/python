import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class VOCSegmentationDataset:
    """PASCAL VOC 2012 세그먼테이션 데이터셋 처리
    
    PASCAL VOC 2012 데이터셋을 세그먼테이션 작업에 맞게 로드하고 전처리하는 클래스
    이미지와 해당하는 세그먼테이션 마스크를 함께 제공함
    """
    
    def __init__(self, root='./data', train=True, download=True):
        """VOC 세그먼테이션 데이터셋 초기화
        """
        # 이미지 전처리: 크기 조정 → 텐서 변환 → ImageNet 기준 정규화
        # 
        # 전처리가 필요한 이유:
        # 1. 배치 처리: 서로 다른 크기의 이미지들을 동일한 크기로 통일
        # 2. 메모리 효율성: 고정 크기로 GPU 메모리 사용량 예측 가능
        # 3. 모델 호환성: CNN은 고정된 입력 크기를 요구
        # 4. 학습 안정성: 정규화로 gradient 폭발/소실 문제 방지
        # 5. 전이학습 효과: ImageNet 사전훈련 모델과 동일한 전처리 적용
        
        self.transform = transforms.Compose([
            # 1단계: 크기 통일 (Resize)
            transforms.Resize((256, 256)),  
            # 적용 이유:
            # - VOC 데이터셋 이미지들은 크기가 제각각 (예: 500x375, 333x500 등)
            # - 배치로 묶기 위해서는 모든 이미지가 같은 크기여야 함
            # - 256x256: 충분한 해상도 + 적당한 연산량의 균형점
            # - GPU 메모리 사용량: 3 × 256 × 256 × batch_size
            
            # 2단계: 데이터 타입 변환 (PIL → Tensor)
            transforms.ToTensor(),  
            # 적용 이유:
            # - PIL Image: uint8 (0~255 정수) → Tensor: float32 (0.0~1.0 실수)
            # - PyTorch 모델은 float 텐서만 처리 가능
            # - 자동으로 (H,W,C) → (C,H,W) 차원 순서 변경
            # - 변환 공식: pixel_value / 255.0
            
            # 3단계: 표준화 (Normalize)
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # 적용 이유:
            # - ImageNet 데이터셋의 RGB 채널별 평균과 표준편차
            # - 평균: [0.485, 0.456, 0.406] (Red, Green, Blue)
            #   0.485 : 빨간색 채널의 평균값
            #   0.456 : 녹색 채널의 평균값
            #   0.406 : 파란색 채널의 평균값
            # - 표준편차: [0.229, 0.224, 0.225] (Red, Green, Blue)
            #   0.229 : 빨간색 채널의 표준편차
            #   0.224 : 녹색 채널의 표준편차
            #   0.225 : 파란색 채널의 표준편차
            # - 변환 공식: (pixel - mean) / std
            # - 결과: 각 채널이 평균 0, 표준편차 1인 분포로 변환
            # - 장점 1: 학습 수렴 속도 향상 (gradient descent 안정화)
            # - 장점 2: ImageNet 사전훈련 모델과 호환 (전이학습 효과)
            # - 장점 3: 각 채널의 스케일 통일로 편향 방지
        ])
        
        # 세그멘테이션 마스크 전처리: 이미지와 다른 특별한 처리 필요
        # 
        # 마스크 처리가 이미지와 다른 이유:
        # 1. 마스크는 클래스 레이블(정수)이지 픽셀 강도가 아님
        # 2. 0=배경, 1=비행기, 2=자전거, ..., 20=TV모니터 (총 21개 클래스)
        # 3. 보간 시 새로운 값 생성하면 안됨 (예: 1.5 클래스는 존재하지 않음)
        # 4. 정규화하면 클래스 정보가 손실됨
        # 5. 이미지와 정확히 같은 크기여야 픽셀별 대응 가능
        
        self.target_transform = transforms.Compose([
            # 마스크 크기 조정: NEAREST 보간법 필수
            transforms.Resize((256, 256), interpolation=Image.NEAREST),
            # NEAREST 사용 이유:
            # - 클래스 레이블 값 보존 (0, 1, 2, ..., 20만 유지)
            # - 부드러운 보간(bilinear, bicubic)은 중간값 생성으로 부적절
            # - 예시: 클래스 1(비행기)과 클래스 2(자전거) 경계에서
            #   bilinear → 1.3, 1.7 같은 의미 없는 값 생성
            #   NEAREST → 1 또는 2만 유지 (올바른 클래스 레이블)
            
            # 마스크를 텐서로 변환
            transforms.ToTensor()
            # 주의사항:
            # - 정규화(Normalize) 하지 않음! 클래스 ID 보존 필요
            # - 결과 텐서: (1, 256, 256) 형태, 값 범위는 0.0~20.0
            # - 각 픽셀값이 해당 위치의 클래스 ID를 나타냄
            # - 손실함수에서 정수로 변환하여 사용: target.long()
        ])
        
        # PASCAL VOC 2012 세그먼테이션 데이터셋 로드
        self.dataset = torchvision.datasets.VOCSegmentation(
            root='data',  # 데이터 저장 경로
            year='2012',  # VOC 2012 버전 사용
            image_set='train' if train else 'val',  # 훈련/검증 데이터셋 선택
            download=False,  # 자동 다운로드 여부
            transform=self.transform,  # 이미지 전처리 함수
            target_transform=self.target_transform  # 마스크 전처리 함수
        )
    
    def __len__(self):
        """데이터셋의 총 샘플 수 반환
        """
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """주어진 인덱스의 데이터 샘플 반환
        """
        return self.dataset[idx]
    
    def get_class_names(self):
        """VOC 데이터셋의 클래스 이름 목록 반환
        
            list: 21개 클래스의 이름 리스트 (배경 포함)
        """
        return [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]

def visualize_segmentation_data():
    """세그먼테이션 데이터 시각화
    
    Purpose: VOC 데이터셋의 이미지와 마스크를 시각적으로 확인하여
    데이터 전처리 결과와 세그먼테이션 품질을 검증
    """
    
    # 1. 데이터셋 로드 (훈련용, 자동 다운로드)
    dataset = VOCSegmentationDataset(train=True, download=True)
    # download=True로 데이터가 없으면 자동 다운로드
    
    # 2. DataLoader 생성: 배치 크기 4, 데이터 셔플링 활성화
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    # batch_size=4: 시각화에 적당한 개수 (2x4 그리드로 표시)
    # shuffle=True: 매번 다른 샘플 확인으로 데이터 다양성 검증

    
    # 3. 첫 번째 배치에서 이미지와 타겟 마스크 추출
    images, targets = next(iter(dataloader))
    # next(iter()) : 전체 데이터 로드 없이 첫 배치만 빠르게 가져옴
    # images: (4, 3, 256, 256) - 4개 이미지, RGB 3채널
    # targets: (4, 1, 256, 256) - 4개 마스크, 1채널 (클래스 ID)
    
    # 4. 시각화를 위한 subplot 생성 (2행 4열)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    # 2행 : 상단=원본이미지, 하단=세그먼테이션마스크 대응 표시
    # 4열 : 배치 크기와 동일 (4개 샘플)
    # figsize=(16,8) : 충분한 크기로 세부사항 확인 가능
    
    for i in range(4):
        # === 이미지 처리 및 시각화 ===
        
        # 5-1. 정규화된 이미지를 원본으로 복원 (ImageNet 정규화 해제)
        img = images[i]  # 배치에서 i번째 이미지 선택: (3, 256, 256)
        
        # 5-2. 표준편차로 곱하기 (역정규화 1단계)
        # 역정규화 하는 이유 : 정규화 과정에서 픽셀값이 0-1 범위로 변환되었기 때문에, 원본 픽셀값으로 복원하기 위해 역정규화 과정을 거침
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        # 정규화 공식 (x-mean)/std 의 역연산
        # view(3,1,1) : 브로드캐스팅으로 (3,256,256)과 연산 가능
        
        # 5-3. 평균값 더하기 (역정규화 2단계)
        # 역정규화 하는 이유 : 정규화 과정에서 픽셀값이 0-1 범위로 변환되었기 때문에, 원본 픽셀값으로 복원하기 위해 역정규화 과정을 거침
        img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        # ImageNet 정규화 완전 해제로 원본 픽셀값 복원
        
        # 5-4. 픽셀값을 0-1 범위로 클리핑
        img = torch.clamp(img, 0, 1)
        # 역정규화 과정에서 생긴 0미만, 1초과 값 제거
        # matplotlib.imshow()는 0-1 범위 요구
        
        # 5-5. 채널 순서 변경: (C, H, W) → (H, W, C)
        img = img.permute(1, 2, 0)
        # PyTorch 텐서 (채널, 높이, 너비) → matplotlib (높이, 너비, 채널)
        # matplotlib.imshow()는 (H,W,C) 형식만 처리 가능
        
        # === 마스크 처리 ===
        
        # 6. 세그먼테이션 마스크 처리
        target = targets[i].squeeze().numpy()
        # squeeze(): (1, 256, 256) → (256, 256) 불필요한 채널 차원 제거  
        # numpy(): matplotlib은 numpy 배열 요구 (torch 텐서 직접 처리 불가)
        
        # === 이미지 표시 ===
        
        # 7-1. 상단 행: 원본 이미지 표시
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Original Image {i+1}')
        axes[0, i].axis('off')  # 축 눈금 제거
        # axis('off'): 픽셀 좌표보다 이미지 내용에 집중하도록
        
        # 7-2. 하단 행: 세그먼테이션 마스크 표시
        axes[1, i].imshow(target, cmap='tab20')
        axes[1, i].set_title(f'Segmentation Mask {i+1}')
        axes[1, i].axis('off')  # 축 눈금 제거
        # cmap='tab20': 
        # - 21개 클래스를 구별 가능한 서로 다른 색상으로 표시
        # - tab20은 20개 구별색 + 기본색으로 21개 클래스 커버
        # - 각 클래스별로 다른 색상으로 영역 구분 명확화
    
    # 8. 레이아웃 조정 및 표시
    plt.tight_layout()  # 서브플롯 간격 자동 조정
    # tight_layout(): 제목과 이미지가 겹치지 않도록 여백 최적화
    
    plt.show()  # 그래프 화면에 표시


# 메인 실행부: 세그먼테이션 데이터 시각화 함수 호출
# Purpose: 스크립트 실행 시 즉시 데이터 시각화를 통해 데이터셋 확인
visualize_segmentation_data()