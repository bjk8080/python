from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.metrics import euclidean_distances
import cv2
import numpy as np
import matplotlib.pyplot as plt

def clustering_segmentation_demo():
    try:
        image=cv2.imread('image.jpg')
        image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        pixel_values=image_rgb.reshape((-1,3))
        pixel_values=np.float32(pixel_values)
        k=5
        kmeans=KMeans(n_clusters=k,random_state=42)
        labels=kmeans.fit_predict(pixel_values)
        centers=np.uint8(kmeans.cluster_centers_)
        segmented_image=centers[labels.flatten()]
        segmented_image=segmented_image.reshape(image_rgb.shape)
        small_image=cv2.resize(image_rgb,(image_rgb.shape[1]//2,image_rgb.shape[0]//2))
        small_pixel_values=small_image.reshape((-1,3))
        small_pixel_values=np.float32(small_pixel_values)

        print(f"Original image size: {image_rgb.shape}")
        print(f"Reduced image size: {small_image.shape}")
        print(f"Pixel count reduced from {len(pixel_values)} to {len(small_pixel_values)}")

        sample_size = min(10000,len(small_pixel_values))
        sample_indices=np.random.choice(len(small_pixel_values),sample_size, replace=False)
        sampled_pixels=small_pixel_values[sample_indices]

        print(f"using{len(sampled_pixels)}sampled pixels for Mean shift")

        mean_shift=MeanShift(bandwidth=30)
        print("Mean shift 클러스터링 시작")
        labels_ms_sample=mean_shift.fit_predict(sampled_pixels)
        print("mean shift 클러스터링 완료!")

        centers_ms=np.uint8(mean_shift.cluster_centers_)
        print(f"Found{len(centers_ms)} clusters")

        distances=euclidean_distances(small_pixel_values,centers_ms)

        labels_ms_full=np.argmin(distances,axis=1)
        segmented_image_ms=centers_ms[labels_ms_full]
        segmented_image_ms=segmented_image_ms.reshape(small_image.shape)

        segmented_image_ms=cv2.resize(segmented_image_ms,(image_rgb.shape[1],image_rgb.shape[0]))

        print(f"segmented_image_ms:{segmented_image_ms.shape}")

        plt.figure(figsize=(15,5))

        plt.subplot(1,3,1)
        plt.imshow(image_rgb)
        plt.title('original Image')
        plt.axis('off')

        plt.subplot(1,3,2)
        plt.imshow(segmented_image)
        plt.title(f'K-menans (k={k})')
        plt.axis('off')

        plt.subplot(1,3,3)
        plt.imshow(segmented_image_ms)
        plt.title('Mean shift (Optimized)')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        print(f"segmented_image: {segmented_image.shape}")

        return segmented_image, segmented_image_ms

    except Exception as e:
        print(f"Error: {e}")
        return None, None
if __name__ == "__main__":
    clustering_segmentation_demo()
