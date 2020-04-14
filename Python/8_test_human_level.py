#!/usr/bin/env python

"""
we use this script to test the human accuracy level of differentiate infarction
"""
import os
import os.path
import shutil
import pandas as pd
import numpy as np
import settings
import function_list as ff
cg = settings.Experiment()

# # step 1: find validation set
# test_patient_list= np.load(os.path.join(cg.nas_main_dir,'test_patient_list.npy'),allow_pickle=True)

# # step 2: pick 10 patients
# L = []
# for i in range(0,10):
#     patient = test_patient_list[i]
#     movie_list = ff.find_all_target_files([patient+'*'],os.path.join(cg.nas_main_dir,'movie'))
#     for m in movie_list:
#         L.append(m)
# L = np.asarray(L)
# print(L.shape)

# # step 3: shuffle the patient
# np.random.seed(1)
# np.random.shuffle(L)

# # step 4: make a excel file to save shuffle result + copy files
# human_folder = os.path.join(cg.nas_main_dir,'human_test')
# ff.make_folder([human_folder])

# result = ()
# for i in range(L.shape[0]):
#     l  = L[i]
#     # copy
#     if os.path.exists(os.path.join(human_folder,'case'+str(i+1)+'.avi')):
#         continue
#     shutil.copyfile(l,os.path.join(human_folder,'case'+str(i+1)+'.avi'))


#     file_name = l.split(os.path.sep)[-1]
#     strain_reduction = float(file_name.split('_')[-2])
#     if strain_reduction >= 0.7:
#         class_name = 'severe'
#         class_no = 2
#     elif strain_reduction <= 0.2:
#         class_name = 'normal'
#         class_no = 0
#     else:
#         class_name = 'mild'
#         class_no = 1
#     r = [i+1,file_name,class_name,class_no,strain_reduction]
#     result = (*result,r)
    

# par = [('Case_No',1),('file_name',1),('class',1),('class_no',1),('strain_reduction',1)]
# excel_save_path = os.path.join(cg.nas_main_dir,'shuffle_list_for_human_test.xlsx')
# ff.xlsx_save(excel_save_path,result,par,list(range(0,5)))

# step 5: include AI validation 
ai = pd.read_csv(os.path.join(cg.nas_main_dir,'lstm3_validation.csv'))
ai_result_list = ai['predict']
ai_file_list = ai['movie_ID']
ai_list = []
for a in ai_file_list:
    ai_list.append(a+'%.avi')
ai_list = np.asarray(ai_list)

human_list = pd.read_csv(os.path.join(cg.nas_main_dir,'shuffle_list_for_human_test.csv'))['file_name']

val_result = ()
for i in range(len(human_list)):
    h = human_list[i]
    index = np.where(ai_list == h)
    val = ai_result_list[index[0][0]]
    r = [i+1, h, val]
    val_result = (*val_result,r)

par = [('Case_No',1),('file_name',1),('AI_result',1)]
excel_save_path = os.path.join(cg.nas_main_dir,'ai_result_for_human_test.xlsx')
ff.xlsx_save(excel_save_path,val_result,par,list(range(0,3)))








