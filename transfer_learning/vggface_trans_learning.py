import os
import csv
import torch
import torchvision
from torchvision import transforms
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dnnbrain.dnn.io import PicDataset
from dnnbrain.dnn.models import dnn_train_model,Vgg_face,dnn_test_model
import torchvision.models as models


###import models

vgg_face = Vgg_face()
vgg_face.load_state_dict(torch.load('/nfs/a1/userhome/xiehailun/workingdir/new_study/dnn/vgg_face_dag.pth'))

###change the fc8 parameters

for param in vgg_face.parameters():
    param.requires_grad = False
in_features = vgg_face.fc8.in_features
vgg_face.fc8 = nn.Linear(in_features,60)


###training process 

params = filter(lambda p: p.requires_grad, vgg_face.parameters())
optimizer = torch.optim.Adam(vgg_face.parameters(),lr = 0.01)
loss_func = nn.CrossEntropyLoss()

###load the data

train_path =  r'/nfs/a1/userhome/xiehailun/workingdir/new_study/face_stimuli/all_train_validation_info.stim.csv'
validation_path = r'/nfs/a1/userhome/xiehailun/workingdir/new_study/face_stimuli/all_vali_info.stim.csv'

###data transforms
trans = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean = [0.481, 0.457, 0.398], 
                             std = [0.237,0.233,0.231])])
                        
                             
train_dataset = PicDataset(train_path,trans)
validation_dataset = PicDataset(validation_path,trans)


train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True) 
validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=True)

###DCNN transfer learning

train_model = dnn_train_model(train_dataloader,vgg_face,loss_func,optimizer,100,'tradition')
torch.save(vgg_face.state_dict(),'/nfs/a1/userhome/xiehailun/workingdir/new_study/dnn/transfer_DCNN/vgg_face.pth')









