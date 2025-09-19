import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def calculate_iou(box1, box2):
    """
    두 바운딩 박스 간의 IoU(Intersection over Union) 계산
    """
    # 교집합 영역의 좌표 계산
    # 두 박스가 겹치는 영역의 좌상단과 우하단 좌표를 구함
    x1_inter = max(box1[0], box2[0])  # 교집합의 왼쪽 x 좌표 (더 큰 값)
    y1_inter = max(box1[1], box2[1])  # 교집합의 위쪽 y 좌표 (더 큰 값)
    x2_inter = min(box1[2], box2[2])  # 교집합의 오른쪽 x 좌표 (더 작은 값)
    y2_inter = min(box1[3], box2[3])  # 교집합의 아래쪽 y 좌표 (더 작은 값)
    
    # 교집합 영역이 존재하지 않는 경우 (박스가 겹치지 않음)
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    # 교집합 영역의 넓이 계산
    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # 각 박스의 넓이 계산
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 합집합 영역의 넓이 계산 (두 박스의 넓이 합 - 교집합 넓이)
    union_area = box1_area + box2_area - intersection_area
    
    # IoU 계산 및 반환
    return intersection_area / union_area

def nms(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression 구현
    """
    
    # 신뢰도 순으로 정렬 (내림차순)
    # argsort()는 오름차순 정렬의 인덱스를 반환하므로 [::-1]로 뒤집어서 내림차순으로 만듦
    indices = np.argsort(scores)[::-1]
    # indices: 신뢰도 순으로 정렬된 박스의 인덱스 배열
    # [::-1]: 내림차순 정렬
    
    # 유지할 박스들의 인덱스를 저장할 리스트
    keep = []
    
    # 모든 박스를 처리할 때까지 반복
    while len(indices) > 0:
        # 가장 높은 신뢰도의 박스 선택 (현재 남은 박스 중에서)
        current = indices[0]
        keep.append(current)  # 이 박스는 유지
        
        # 마지막 박스인 경우 종료
        if len(indices) == 1:
            break
        
        # 현재 박스와 나머지 박스들의 IoU 계산
        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]  # 현재 박스를 제외한 나머지 박스들
        
        # 각 박스와의 IoU 계산
        ious = []
        for other_box in other_boxes:
            iou = calculate_iou(current_box, other_box)
            ious.append(iou)
        
        ious = np.array(ious)
        
        # IoU가 임계값보다 작은 박스들만 남김
        # 임계값보다 큰 박스들은 현재 박스와 중복으로 간주하여 제거
        # 
        # 예시: IoU 임계값이 0.5인 경우
        # - IoU > 0.5: 많이 겹침 → 같은 객체의 중복 검출 → 제거
        # - IoU < 0.5: 적게 겹침 → 다른 객체 검출 → 유지
        #
        # 왜 낮은 IoU만 남기는가?
        # 1. 높은 IoU = 같은 객체를 여러 번 검출한 것 = 불필요한 중복
        # 2. 낮은 IoU = 서로 다른 객체를 검출한 것 = 유지해야 함
        # 3. 신뢰도가 가장 높은 박스는 이미 keep에 추가했으므로
        #    겹치는 다른 박스들은 제거하는 것이 올바름
        indices = indices[1:][ious < iou_threshold]
    
    return keep

def visualize_nms():
    """
    NMS 과정 시각화
    """
    
    # 샘플 바운딩 박스와 신뢰도
    # 5개의 박스 중 일부는 서로 겹치도록 설정하여 NMS 효과를 보여줌
    boxes = np.array([
        [50, 50, 150, 150],    # 박스 1: 좌상단 (50,50), 우하단 (150,150)
        [60, 60, 160, 160],    # 박스 2: 박스 1과 겹침, 약간 오른쪽 아래로 이동
        [200, 100, 300, 200], # 박스 3: 별도 위치
        [210, 110, 310, 210], # 박스 4: 박스 3과 겹침, 약간 오른쪽 아래로 이동
        [70, 70, 170, 170]    # 박스 5: 박스 1, 2와 겹침, 더 오른쪽 아래로 이동
    ])
    
    # 각 박스의 신뢰도 점수 (0~1 사이의 값)
    # 박스 1이 가장 높은 신뢰도를 가짐
    scores = np.array([0.9, 0.8, 0.85, 0.7, 0.75])
    # 신뢰도 정보를 좀 더 말 되게 변경 
    
    # NMS 적용 (IoU 임계값 0.3으로 설정 - 상당히 엄격한 기준)
    keep_indices = nms(boxes, scores, iou_threshold=0.3)
    
    # 시각화를 위한 subplot 생성 (1행 2열)
    plt.rc('font', family='Malgun Gothic')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # NMS 적용 전 (왼쪽 그래프)
    # 배경 이미지 생성 (회색 배경)
    img = np.ones((350, 400, 3)) * 0.9
    axes[0].imshow(img)
    axes[0].set_title('NMS 적용 전')
    
    # 각 박스를 구분하기 위한 색상 배열
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # 모든 박스를 그리기
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box
        # 사각형 패치 생성 (테두리와 반투명 채우기)
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=2, edgecolor=colors[i], 
                               facecolor=colors[i], alpha=0.3)
        axes[0].add_patch(rect)
        # 박스 번호와 신뢰도 점수 표시
        axes[0].text(x1, y1-5, f'Box {i+1}: {score:.2f}', 
                    color=colors[i], fontsize=10, weight='bold')
    
    # 축 설정
    axes[0].set_xlim(0, 400)
    axes[0].set_ylim(350, 0)  # y축 뒤집기 (이미지 좌표계)
    axes[0].axis('off')  # 축 눈금 제거
    
    # NMS 적용 후 (오른쪽 그래프)
    axes[1].imshow(img)
    axes[1].set_title('NMS 적용 후')
    
    # 유지된 박스들만 그리기
    for i in keep_indices:
        box, score = boxes[i], scores[i]
        x1, y1, x2, y2 = box
        # 유지된 박스는 더 두꺼운 테두리로 강조
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=3, edgecolor=colors[i], 
                               facecolor=colors[i], alpha=0.3)
        axes[1].add_patch(rect)
        axes[1].text(x1, y1-5, f'Box {i+1}: {score:.2f}', 
                    color=colors[i], fontsize=10, weight='bold')
    
    axes[1].set_xlim(0, 400)
    axes[1].set_ylim(350, 0)
    axes[1].axis('off')
    
    # 레이아웃 조정 및 그래프 표시
    plt.tight_layout()
    plt.show()
    
    # NMS 결과 통계 출력
    print(f"원본 박스 수: {len(boxes)}")
    print(f"NMS 후 박스 수: {len(keep_indices)}")
    print(f"제거된 박스: {[i+1 for i in range(len(boxes)) if i not in keep_indices]}")


visualize_nms()