import numpy as np  # 수치 계산을 위한 라이브러리
import matplotlib.pyplot as plt  # 그래프 시각화를 위한 라이브러리
from sklearn.metrics import precision_recall_curve, average_precision_score  
# 성능 평가 메트릭 계산을 위한 라이브러리

def calculate_map_example():
    """
    mAP (mean Average Precision) 계산 예제 함수
    """
    
    # 가상의 검출 결과 (실제로는 모델 예측 결과)
    np.random.seed(42)  # 재현 가능한 결과를 위한 시드 설정
    # 실행할 때마다 동일한 결과를 얻기 위해 랜덤 시드를 고정
    
    # 3개 클래스에 대한 예제 데이터
    classes = ['person', 'car', 'bicycle']  # 검출할 객체 클래스 목록
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # 2행 3열의 서브플롯 생성
    # 매개변수 설명:
    # - 2, 3: 2행 3열 배치 (각 클래스당 2개의 그래프)
    # - figsize=(18, 10): 전체 그림의 크기 (가로 18인치, 세로 10인치)
    # 3개 클래스 각각에 대해 PR 곡선과 임계값 그래프를 동시에 표시하기 위함
    
    total_ap = 0  # 전체 AP의 합을 저장할 변수
    #  mAP 계산을 위해 모든 클래스의 AP를 누적할 변수
    
    for i, class_name in enumerate(classes):  # 각 클래스별로 반복 처리
        # enumerate(): 인덱스와 값을 동시에 반환하는 함수
        # 클래스 인덱스(i)와 클래스명(class_name)을 모두 사용하기 위함
        
        # 가상의 ground truth와 예측 결과
        n_samples = 100  # 샘플 개수
        # 각 클래스별로 100개의 테스트 샘플을 시뮬레이션
        
        y_true = np.random.randint(0, 2, n_samples)  # 0 또는 1의 실제 라벨
        # 매개변수 설명:
        # - 0, 2: 0 이상 2 미만의 정수 (즉, 0 또는 1)
        # - n_samples: 생성할 샘플 개수
        # 이진 분류 문제에서 실제 정답 라벨 (0: 해당 클래스 없음, 1: 해당 클래스 있음)
        
        y_scores = np.random.random(n_samples)  # 신뢰도 점수 (0~1 사이)
        # 설명: 모델이 예측한 신뢰도 점수를 시뮬레이션 (높을수록 해당 클래스일 확률이 높음)
        
        # 더 현실적인 데이터를 위한 조정
        # 실제 양성 샘플에는 더 높은 점수를 부여
        y_scores[y_true == 1] += 0.3  # 실제 양성 샘플의 신뢰도를 0.3만큼 증가
        # 
        
        y_scores = np.clip(y_scores, 0, 1)  # 0~1 범위로 제한
        # 매개변수 설명:
        # - y_scores: 클리핑할 배열
        # - 0, 1: 최소값과 최대값
        # 신뢰도 점수가 1을 초과하지 않도록 제한
        
        # Precision-Recall 곡선 계산
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        # 매개변수 설명:
        # - y_true: 실제 라벨
        # - y_scores: 예측 신뢰도 점수
        # 반환값:
        # - precision: 각 임계값에서의 정밀도
        # - recall: 각 임계값에서의 재현율
        # - thresholds: 임계값 배열
        # PR 곡선을 그리기 위한 데이터 생성
        
        ap = average_precision_score(y_true, y_scores)  # Average Precision 계산
        # PR 곡선 아래 면적을 계산하여 해당 클래스의 성능을 하나의 수치로 표현
        
        # PR 곡선 시각화
        axes[0, i].plot(recall, precision, linewidth=2, label=f'AP = {ap:.3f}')
        # 매개변수 설명:
        # - recall, precision: x축, y축 데이터
        # - linewidth=2: 선의 두께
        # - label: 범례에 표시될 텍스트 (AP 값 포함)
        #  Precision-Recall 관계를 시각적으로 표현
        
        axes[0, i].fill_between(recall, precision, alpha=0.3)  # 곡선 아래 영역 채우기
        # 매개변수 설명:
        # - recall, precision: 채울 영역의 경계
        # - alpha=0.3: 투명도 (0.3으로 설정하여 반투명하게)
        #  AP 값(곡선 아래 면적)을 시각적으로 강조
        
        axes[0, i].set_xlabel('Recall')  # x축 라벨
        axes[0, i].set_ylabel('Precision')  # y축 라벨
        axes[0, i].set_title(f'{class_name} - PR Curve')  # 그래프 제목
        axes[0, i].legend()  # 범례 표시
        axes[0, i].grid(True, alpha=0.3)  # 격자 표시 (투명도 0.3)
        axes[0, i].set_xlim([0, 1])  # x축 범위 설정
        axes[0, i].set_ylim([0, 1])  # y축 범위 설정
        # 그래프의 가독성을 높이고 표준화된 형태로 표시
        
        # 신뢰도 임계값에 따른 정밀도/재현율 변화
        axes[1, i].plot(thresholds, precision[:-1], 'b-', label='Precision')
        # 매개변수 설명:
        # - thresholds: x축 (신뢰도 임계값)
        # - precision[:-1]: y축 (정밀도, 마지막 원소 제외)
        # 마지막 원소 제외 이유 : 마지막 원소는 1.0이므로 그래프가 완벽한 직선이 되어버림
        # - 'b-': 파란색 실선
        # precision 배열이 thresholds보다 1개 많으므로 마지막 원소 제외
        
        axes[1, i].plot(thresholds, recall[:-1], 'r-', label='Recall')
        # 매개변수 설명:
        # - 'r-': 빨간색 실선
        # 임계값 변화에 따른 정밀도와 재현율의 트레이드오프 관계 시각화
        
        axes[1, i].set_xlabel('Confidence Threshold')  # x축: 신뢰도 임계값
        axes[1, i].set_ylabel('Score')  # y축: 점수
        axes[1, i].set_title(f'{class_name} - Precision vs Recall')  # 그래프 제목
        axes[1, i].legend()  # 범례 표시
        axes[1, i].grid(True, alpha=0.3)  # 격자 표시
        axes[1, i].set_xlim([0, 1])  # x축 범위
        axes[1, i].set_ylim([0, 1])  # y축 범위
        # 임계값 조정이 성능에 미치는 영향을 시각화
        
        total_ap += ap  # 현재 클래스의 AP를 총합에 추가
        # mAP 계산을 위해 모든 클래스의 AP를 누적
    
    # mAP 계산
    map_score = total_ap / len(classes)  # 평균 계산
    # mean Average Precision = (모든 클래스의 AP 합) / (클래스 개수)
    
    plt.suptitle(f'mAP = {map_score:.3f}', fontsize=16, fontweight='bold')
    # 매개변수 설명:
    # - f'mAP = {map_score:.3f}': 제목 텍스트 (소수점 3자리까지 표시)
    # - fontsize=16: 글꼴 크기
    # - fontweight='bold': 굵은 글꼴
    # 전체 그래프의 제목으로 mAP 값을 명확히 표시
    
    plt.tight_layout()  # 서브플롯 간격 자동 조정
    # 그래프들이 겹치지 않도록 레이아웃 최적화
    
    plt.show()  # 그래프 화면에 출력
    
    # 클래스별 AP 출력
    print("클래스별 Average Precision:")
    for i, class_name in enumerate(classes):
        print(f"  {class_name}: {total_ap/len(classes):.3f}")  # 실제로는 각각 다른 값
        # 주의: 현재 코드는 모든 클래스에 대해 동일한 값(mAP)을 출력
        # 실제로는 각 클래스별로 계산된 개별 AP 값을 저장하여 출력해야 함
    print(f"\nmAP: {map_score:.3f}")
    # 

# 함수 실행
calculate_map_example()
# 정의된 함수를 실행하여 mAP 계산 예제를 실행