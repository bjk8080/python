import cv2
import numpy as np
import matplotlib.pyplot as plt

def threshold_sgmentation_demo():
    image=cv2.imread('image.jpg',0)
    ret, binary=cv2.threshold(image,127,255,cv2.THRESH_BINARY)
    ret, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(image, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    adative=cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    
    plt.figure(figsize=(15,5))

    plt.subplot(1,4,1)
    plt.imshow(image, cmap='gray')
    plt.title('original Image')
    plt.axis('off')

    plt.subplot(1,4,2)
    plt.imshow(binary, cmap='gray')
    plt.title('Binary Threshold')
    plt.axis('off')

    plt.subplot(1,4,3)
    plt.imshow(otsu, cmap='gray')
    plt.title("otsu's Method")
    plt.axis('off')

    plt.subplot(1,4,4)
    plt.imshow(adaptive, cmap='gray')
    plt.title('adaptive Threshold')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return binary, otsu, adaptive

threshold_sgmentation_demo()