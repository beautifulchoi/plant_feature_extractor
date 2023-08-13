from inferencer import Inferencer
import visualization as vi
import torch
from torch import nn
from preprocessing import Preprocessing
from model import EfficientB4
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import os

def get_CAM(path,use_gpu=False,shape=(224,224),type=1, method='eigenCAM', save_name='sample_data'):
    if use_gpu==True:
        if torch.cuda.is_available():
            device="cuda"
    else:
        device="cpu"
        
    
    print(f"Using {device} device")
    
    original_shape=np.array(Image.open(path)).shape
    original=plt.imshow(Image.open(path))
    plt.title("origianl image")
    plt.axis('off')
    plt.show(original)
    
    #load model
    model = EfficientB4(1081, loss_fn=nn.CrossEntropyLoss(), fc_type="shallow")
    pt_path=os.path.join(os.getcwd(),'exp_ckpt/checkpoints/checkpoint.pt')
    model.load(pt_path)
    model=model.to(device)
    
    #put model and get similar plant output
    infer=Inferencer(model,path,device)
    prob, idx=infer()
    
    #resizing
    preprocess=Preprocessing(path)
    resized_img=preprocess.resize_img(shape)
    
    if type==1:
        cam=vi.printCAM(idx,resized_img,model, [model.patch_embedding],device=device)
        title="various CAM method output"
        fig = plt.figure()
        plt.title(title,y=1.3,fontsize=15)
        plt.axis('off')
        plt.imshow(vi.make_table(cam, (1,len(cam))))
        
        word=['original','gradCAM','layerCAM','eigenCAM','gradCAM++','scoreCAM']
        for i in range(6):
            plt.text(shape[0]*i,-10,word[i])
        plt.show()
        fig.savefig('{}.jpg'.format(save_name),format='jpg', dpi=300)
    elif type==2:
        cam, point=vi.diverse_CAM(idx,resized_img,model, method, [model.patch_embedding],device=device)
        title="{} output".format(method)
        fig = plt.figure()
        plt.title(title,y=1.3,fontsize=15)
        plt.axis('off')
        plt.imshow(vi.make_table(cam, (1,len(cam))))
        word=['original','CAM(gray)','CAM(color)',method,'detection box']
        for i in range(5):
            plt.text(shape[0]*i,-10,word[i])
        plt.show()
        fig.savefig('{}.jpg'.format(save_name).format(2,method,path.split('/')[-2]),format='jpg', dpi=300)
        
        new_points=[]
        for p in point:
            new_x,new_y=p[0]*(original_shape[1]/shape[1]),p[1]*(original_shape[0]/shape[0]) #리사이즈 전 원본좌표 반환
            new_points.append((new_x,new_y))  
        
        return point, tuple(new_points)
    
    