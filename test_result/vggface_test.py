import os
import csv
import torch
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
from dnnbrain.dnn.io import PicDataset
from dnnbrain.dnn.models import Vgg_face,dnn_test_model
import pandas as pd


###import models

vgg_face = Vgg_face()
for param in vgg_face.parameters():
    param.requires_grad = False
in_features = vgg_face.fc8.in_features
vgg_face.fc8 = nn.Linear(in_features,60)
vgg_face.load_state_dict(torch.load('/nfs/a1/userhome/xiehailun/workingdir/new_study/dnn/transfer_DCNN/vggface.pth'))

###load test or rotation inputs

test_path = r'/nfs/a1/userhome/xiehailun/workingdir/new_study/face_stimuli/all_test_info.stim.csv'

### or change or the 'test' into 'rotation' to get the results of inverted inputs

trans = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean = [0.485,0.456,0.406], 
                             std = [0.229, 0.224, 0.225])])
test_dataset = PicDataset(test_path,trans)
test_dataloader = DataLoader(test_dataset, batch_size=64) 
model_target, actual_target, test_acc = dnn_test_model(test_dataloader,vgg_face)

###export results

df1 = pd.DataFrame(actual_target)
df1.to_csv('/nfs/a1/userhome/xiehailun/workingdir/new_study/result/vggface_test_input.csv',index = False)
df2 = pd.DataFrame(model_target)
df2.to_csv('/nfs/a1/userhome/xiehailun/workingdir/new_study/result/vggface_test_output.csv',index = False)




