import torch 
import json
from torch import nn
from preprocessing import Preprocessing
import os

class Inferencer(): #사진에 대한 top1 acc 정보를 plantnet 기반으로 가져옴
    
    def __init__(self, model,path,device="cpu"): 
        #categories is total sorted categories' list of data(must have to put labeled one)
        self.model=model
        self.device=device
        self.path=path 
        
        now_path=os.getcwd()
        json_path=os.path.join(now_path,"plantnet300K_species_id_2_name.json")
        with open(json_path, 'r') as f:
            self.json_data = json.load(f)
    
        self.categories=self.get_categories()
    
    def get_categories(self):
        categories=sorted(list(self.json_data.keys()))
        return categories
    
    def __call__(self): #객체 호출하면 바로 예측 top 1 확률값과  값이 나오도록
       
        

        preprocessing=Preprocessing(self.path)
        X=preprocessing.apply_transform()
            
        self.model.eval()    
        with torch.no_grad():
            pred=self.model(X.to(self.device))
            prob=nn.Softmax(dim=1)
            pred=prob(pred)
            predicted_top5=torch.topk(pred,1)    #한 배치의 top1정보 지님, 형태는 [value= batch size*5 형태, indices=b size*5 형태]
            batch_scores=predicted_top5[0] #probability of top 1 (by softmax)
            batch_indices=predicted_top5[1]# index of that plant
        
        
        #name_list=self.label2name()
        top1_info=(batch_scores[0][0].item(),batch_indices[0][0].item())
                
        return top1_info
    
    # def label2name(self):
    #     #카테고리명이 라벨로 되어있는 경우 식물이름으로 변환시켜주기
    #     name_list=[]
    #     for label in self.categories:
    #         plantname=self.json_data[label]
    #         name_list.append(plantname)

    #     return name_list

        
    