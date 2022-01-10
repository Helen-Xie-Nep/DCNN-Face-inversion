from dnnbrain.dnn.core import Activation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#fig = plt.figure()

activ1 = Activation()
activ1.load(r'/nfs/a1/userhome/xiehailun/workingdir/new_study/result/vggface_test_fc3.act.h5')
vggface_upright = activ1.get('fc8')

face_upright,object_upright = np.split(vggface_upright,2,axis = 0)

activ2 = Activation()
activ2.load(r'/nfs/a1/userhome/xiehailun/workingdir/new_study/result/vggface_rotation_fc3.act.h5')
vggface_inverted = activ2.get('fc8')
face_inverted,object_inverted = np.split(vggface_inverted,2,axis = 0)


vggface_activation = np.vstack((face_upright,face_inverted,object_upright,object_inverted))
vggface_activation = vggface_activation.reshape([6000,60])
vggface_activation = pd.DataFrame(vggface_activation)
all_average = pd.DataFrame()
for i in range(0,120):
    every_ID = vggface_activation[i*50:(i+1)*50]
    
    mean_ID = every_ID.mean()
    all_average = all_average.append(mean_ID, ignore_index=True)

all_average = all_average.T
average_corr = all_average.corr()

#save out
np.savetxt('/nfs/a1/userhome/xiehailun/workingdir/new_study/new_result/new_vggface_fc3.csv',average_corr,delimiter=',')





