#get alexnet fc3 activation as an example,also could change "fc3" to other layers to get the activation

import os
import torch
from torch import nn
from os.path import join as pjoin
from dnnbrain.dnn.core import Stimulus, Mask
from dnnbrain.dnn.models import AlexNet

#Load the network
fc3_new = nn.Linear(in_features = 4096,out_features = 60, bias = True)
dnn = AlexNet()
dnn.model.classifier[6] = fc3_new
dnn.model.load_state_dict(torch.load('/nfs/a1/userhome/xiehailun/workingdir/new_study/dnn/transfer_DCNN/alexnet.pth'))

DNNBRAIN_TEST = pjoin(os.environ['DNNBRAIN_DATA'], 'test')
TMP_DIR = pjoin(os.environ['HOME'], '.dnnbrain_tmp')
if not os.path.isdir(TMP_DIR):
    os.makedirs(TMP_DIR)

# Load stimuli information
stim_file = pjoin(DNNBRAIN_TEST, 'image', '/nfs/a1/userhome/xiehailun/workingdir/new_study/face_stimuli/all_test_info.stim.csv',)
#could also change "test" to "rotation" to get the activation of inverted stimuli
stimuli = Stimulus()
stimuli.load(stim_file)

# Load mask information

dmask = Mask()
dmask.set('fc3')

# Extract DNN activation

activation = dnn.compute_activation(stimuli, dmask)

# Save out (fc3 as "fc8" in this condition)
out_file = pjoin(TMP_DIR, '/nfs/a1/userhome/xiehailun/workingdir/new_study/result/alexnet_test_fc8.act.h5')
activation.save(out_file)
