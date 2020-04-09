#!/usr/bin/env python

import glob
import os
import os.path
import numpy as np
import settings
import function_list as ff
import pandas as pd
from sklearn.model_selection import train_test_split
cg = settings.Experiment() 

# siemens 304225 is not in the infarction movie set
patient_list = ff.find_all_target_files(['ucsd_bivent/*','ucsd_lvad/*','ucsd_ccta/*','ucsd_toshiba/*','ucsd_tavr_1/*','ucsd_pv/*','ucsd_siemens/2*'],os.path.join(cg.nas_patient_dir))
print(patient_list.shape)

np.random.shuffle(patient_list)
a = np.array_split(patient_list,10)

# split train and test by 7:3
train = []
for i in range(0,7):
    aa = a[i]
    for j in aa:
        parts = j.split(os.path.sep)
        patient_name = parts[-2]+'_'+parts[-1]
        train.append(patient_name)
      
        
train = np.asarray(train)

test = []
for i in range(7,10):
    aa = a[i]
    for j in aa:
        parts = j.split(os.path.sep)
        patient_name = parts[-2]+'_'+parts[-1]
        test.append(patient_name)
test = np.asarray(test)

np.save(os.path.join(cg.nas_main_dir,'train_patient_list'),train)
np.save(os.path.join(cg.nas_main_dir,'test_patient_list'),test)


