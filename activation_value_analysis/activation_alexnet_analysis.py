from dnnbrain.dnn.core import Activation
import numpy as np
import pandas as pd


#input the activation value
activ1 = Activation()
activ1.load(r'/nfs/a1/userhome/xiehailun/workingdir/new_study/result/alexnet_test_fc3.act.h5')
alexnet_upright = activ1.get('fc3')

face_upright,object_upright = np.split(alexnet_upright,2,axis = 0)

activ2 = Activation()
activ2.load(r'/nfs/a1/userhome/xiehailun/workingdir/new_study/result/alexnet_rotation_fc3.act.h5')
alexnet_inverted = activ2.get('fc3')
face_inverted,object_inverted = np.split(alexnet_inverted,2,axis = 0)

#activation value analysis
alexnet_activation = np.vstack((face_upright,face_inverted,object_upright,object_inverted))
alexnet_activation = alexnet_activation.reshape([6000,60])
alexnet_activation = pd.DataFrame(alexnet_activation)
all_average = pd.DataFrame()
for i in range(0,120):
    every_ID = alexnet_activation[i*50:(i+1)*50]
    
    mean_ID = every_ID.mean()
    all_average = all_average.append(mean_ID, ignore_index=True)

all_average = all_average.T
average_corr = all_average.corr()

#save out
np.savetxt('/nfs/a1/userhome/xiehailun/workingdir/new_study/new_result/new_alexnet_fc3.csv',average_corr,delimiter=',')


