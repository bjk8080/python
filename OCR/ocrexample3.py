import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt



class OCRPreprocessor:
    def __init__(self):
        pass

    def convert_to_grayscale(self, image):
        
        if len(image.shape)==3:
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        else:
            gray=image.copy()
        return gray
    def apply_threshold(self,image,method='adaptive'):
        gray=self.convert_to_grayscale(image)
        if method=='simple':
            _, thresh=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
        elif method== 'adaptive':
            thresh=cv2.adaptiveThreshold(
                gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,11,2
            )
        elif method=='otsu':
            _,thresh=cv2.threshold(
                gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU
            )

        return thresh
    
    def remove_noise(self,image):
        kernel=np.ones((3,3),np.uint8)
        opening=cv2.morphologyEx(image, cv2.MORPH_OPEN,kernel)
        closing=cv2.mrphologyEx(opening,cv2.MORPH_CLOSE,kernel)
        denoised=cv2.GaussianBlur(closing,(3,3),0)

    def correct_skew(self,image):
        edges=cv2.Canny(image,50,150,apertureSize=3)
        lines=cv2.HoughLines(edges,1,np.pi/180,threshold=100)
    
        if lines is not None:
            angles=[]
            for rho, theta in lines[:,0]:
                angle=np.degrees(theta)-90
                angles.append(angle)
                median_angle=np.median(angles)

                (h,w)=image.shape[:2]
                center=(w//2,h//2)
                M=cv2.getRotationMatrix2D(center,median_angle,1.0)
                rotated=cv2.warpAffine(image,M,(w,h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE)
                
                return rotated
        return image
    
    def resize_image(self,image, target_height=800):
        h,w = image.shape[:2]

        if h< target_height:
            scale=target_height/h
            scale=target_height/h
            new_w=int(w*scale)

            resized=cv2.resize(image,(new_w,target_height),interpolation=cv2.INTER_CUBIC)
        else:
            resized=image
        
        return resized
    
    def visualize_preprocessing_steps(self,steps,step_names):
        plt.rc('font',family='malgun gothic')
        fig,axes=plt.subplots(2,3,figsize=(15,10))
        axes=axes.ravel()

        for i, (step,name) in enumerate(zip(steps,step_names)):
            if i < len(axes):
                if len(step.shape)==3:
                    axes[i].imshow(cv2.cvtColor(step,cv2.COLOR_BGR2RGB))
                else:
                    axes[i].imshow(step,cmap='gray')

                axes[i].set_title(name)
                axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def preprocess_pipeline(self,image,visualize=False):

        steps=[]
        step_names


