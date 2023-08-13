import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image


class Preprocessing:
    def __init__(self,path):
        self.path=path
        
    def resize_img(self,img_shape=None): #resizing img to img_shape
        images = []
        img=self.bring_img()
        if img_shape!=None:
            img = cv2.resize(img,img_shape)
        img = np.float32(img) / 255
        images.append(img)
        return np.array(images) #(380,380,3) 인 이미지들들을 리스트로 묶어서 반환시킴

    def transform(self):
        data_transforms = A.Compose([
        A.LongestMaxSize(max_size=500),
        A.PadIfNeeded(min_height=int(380),
        min_width=int(380),
        position='top_left',
        border_mode=cv2.BORDER_CONSTANT),
        A.CenterCrop(height=380, width=380, p=1.0),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()])
        return data_transforms
        
    def bring_img(self): 
            
        image = np.array(Image.open(self.path))
        if image.shape[2]>3: # if more than 3 channel (espicially png file) -> just bring 3 channel
            image=image[...:3]
            
        return image #np.array
        
    def apply_transform(self):
        image=self.bring_img()    
        transform=self.transform()
        image = transform(image=image)['image']
                        
        image = image.unsqueeze(0)
        
        return image
    
