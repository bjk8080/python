import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def visualize_bounding_box_formats():
    """바운딩 박스 형식 시각화"""
    
    plt.rc('font', family='Malgun Gothic')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 샘플 이미지 생성 (실제로는 이미지를 로드)
    img = np.ones((300, 400, 3)) * 0.8
    
    # (x1, y1, x2, y2) 형식
    x1, y1, x2, y2 = 50, 50, 200, 150
    
    axes[0].imshow(img)
    rect1 = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                             linewidth=2, edgecolor='red', facecolor='none')
    axes[0].add_patch(rect1)
    axes[0].text(x1, y1-5, f'({x1}, {y1})', color='red', fontsize=10)
    axes[0].text(x2, y2+15, f'({x2}, {y2})', color='red', fontsize=10)
    axes[0].set_title('(x1, y1, x2, y2) 형식')
    axes[0].set_xlim(0, 400)
    axes[0].set_ylim(300, 0)
    
    # (x, y, width, height) 형식
    x, y, w, h = 50, 50, 150, 100
    
    axes[1].imshow(img)
    rect2 = patches.Rectangle((x, y), w, h, 
                             linewidth=2, edgecolor='blue', facecolor='none')
    axes[1].add_patch(rect2)
    axes[1].text(x, y-5, f'({x}, {y})', color='blue', fontsize=10)
    axes[1].text(x+w//2, y+h//2, f'w={w}, h={h}', color='blue', fontsize=10, ha='center')
    axes[1].set_title('(x, y, width, height) 형식')
    axes[1].set_xlim(0, 400)
    axes[1].set_ylim(300, 0)
    
    plt.tight_layout()
    plt.show()

visualize_bounding_box_formats()