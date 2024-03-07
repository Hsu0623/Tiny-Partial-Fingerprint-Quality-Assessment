"""
This code is to retraining on your datasets
"""
from features import get_features
from glob import glob
import os
import cv2
import numpy as np


def statistic_basic_vals(values_list, basic_vals):    
    values_list = np.asarray(values_list)
    a = np.percentile(values_list, 20)
    basic_vals.append(a)
    b = np.percentile(values_list, 40)
    basic_vals.append(b)
    c = np.percentile(values_list, 60)
    basic_vals.append(c)
    d = np.percentile(values_list, 80)
    basic_vals.append(d)
    e = np.max(values_list)
    basic_vals.append(e)

########## modify the path to match your configuration.##############
data_root = [f'', f'']
save_dir = f''       



os.makedirs(save_dir, exist_ok=True)

rsi_means = []
rsi_maxs = []
b_stds = []
b_ocls = []
b_rdis = []

rsi_mean_vals = []
rsi_max_vals = []
b_std_vals = []
b_ocl_vals = []
b_rdis_vals = []

img_path_list = []
for data_root_tmp in data_root:
    img_path_list.extend(sorted(glob(os.path.join(data_root_tmp, '**', '*.bmp'), recursive=True))) 
    
print(f'totoal number of image: {len(img_path_list)}')    
print(f'extracting features from all images.')
for idx, img_path in enumerate(img_path_list):
    
    img = cv2.imread(img_path, 0)
    

    # compute feature of images
    rsi_mean, rsi_max, b_std, b_ocl, b_rdi = get_features(img)
    
    # store
    rsi_means.append(rsi_mean) 
    rsi_maxs.append(rsi_max) 
    b_stds.append(b_std) 
    b_ocls.append(b_ocl) 
    b_rdis.append(b_rdi)

    if idx % (200) == 0:
        print('\r%08d'%idx, end=' ')

rsi_means = np.array(rsi_means)
rsi_maxs = np.array(rsi_maxs)
b_stds = np.array(b_stds) 
b_ocls = np.array(b_ocls)
b_rdis = np.array(b_rdis)

"""
# store features of all images, respectively
np.save(os.path.join(save_dir, 'rsi_means.npy'), rsi_means)
np.save(os.path.join(save_dir, 'rsi_maxs.npy'), rsi_maxs)
np.save(os.path.join(save_dir, 'b_stds.npy'), b_stds)
np.save(os.path.join(save_dir, 'b_ocls.npy'), b_ocls)
np.save(os.path.join(save_dir, 'b_rdis.npy'), b_rdis)
"""

print('rsi_mean')
statistic_basic_vals(rsi_means, rsi_mean_vals)
print(rsi_mean_vals)

print('rsi_max')
statistic_basic_vals(rsi_maxs, rsi_max_vals)
print(rsi_max_vals)

print('block_std')
statistic_basic_vals(b_stds, b_std_vals)
print(b_std_vals) 

print('block_ocl')
statistic_basic_vals(b_ocls, b_ocl_vals)    
print(b_ocl_vals)

print('block_rdi')
statistic_basic_vals(b_rdis, b_rdis_vals)    
print(b_rdis_vals)

with open(os.path.join(save_dir, 'baseline.txt'), 'w') as f:
    f.writelines(' '.join(str(format(f)) for f in rsi_mean_vals))
    f.writelines('\n')
    f.writelines(' '.join(str(format(f)) for f in rsi_max_vals))
    f.writelines('\n')
    f.writelines(' '.join(str(format(f)) for f in b_std_vals))
    f.writelines('\n')
    f.writelines(' '.join(str(format(f)) for f in b_ocl_vals))
    f.writelines('\n')
    f.writelines(' '.join(str(format(f)) for f in b_rdis_vals))
    f.writelines('\n')
    
baselines = []

with open(os.path.join(save_dir, 'baseline.txt'), 'r') as f:
    for line in f:
        values = line.split()
        baselines.append([float(val) for val in values])

baselines = np.array(baselines)   

total = len(img_path_list)
num5 = 0
num4 = 0
num3 = 0
num2 = 0
num1 = 0

basic_features = []
labels = []



for i, img_path in enumerate(img_path_list):
    rsi_mean = rsi_means[i] 
    rsi_max = rsi_maxs[i]
    b_std = b_stds[i]
    b_ocl = b_ocls[i]
    b_rdi = b_rdis[i]
    
    img_name = img_path.split('\\')[-1]
    
    if ((rsi_mean > baselines[0][3]) and (rsi_max > baselines[1][3]) and
       (b_std < baselines[2][0]) and (b_ocl > baselines [3][3]) and (b_rdi > baselines [4][3])):  
        num5 = num5+1
        basic_features.append([rsi_mean, rsi_max, b_std, b_ocl, b_rdi])
        labels.append(4)
        
    elif ((baselines[0][3] >= rsi_mean > baselines[0][2]) and (baselines[1][3] >= rsi_max > baselines[1][2]) and
       (baselines[2][0] <= b_std < baselines[2][1]) and (baselines[3][3] >= b_ocl > baselines [3][2]) and (baselines[4][3] >= b_rdi > baselines [4][2])):
        num4 = num4+1
        basic_features.append([rsi_mean, rsi_max, b_std, b_ocl, b_rdi])
        labels.append(3)
        
    elif ((baselines[0][2] >= rsi_mean > baselines[0][1]) and (baselines[1][2] >= rsi_max > baselines[1][1]) and
       (baselines[2][1] <= b_std < baselines[2][2]) and (baselines[3][2] >= b_ocl > baselines [3][1]) and (baselines[4][2] >= b_rdi > baselines [4][1])):
        num3 = num3+1
        basic_features.append([rsi_mean, rsi_max, b_std, b_ocl, b_rdi])
        labels.append(2)
        
    elif ((baselines[0][1] >= rsi_mean > baselines[0][0]) and (baselines[1][1] >= rsi_max > baselines[1][0]) and
       (baselines[2][2] <= b_std < baselines[2][3]) and (baselines[3][1] >= b_ocl > baselines [3][0]) and (baselines[4][1] >= b_rdi > baselines [4][0])):  
        num2 = num2+1
        basic_features.append([rsi_mean, rsi_max, b_std, b_ocl, b_rdi])
        labels.append(1)
        
    elif ((baselines[0][0] >= rsi_mean) and (baselines[1][0] >= rsi_max) and
       (baselines[2][3] <= b_std) and (baselines[3][0] >= b_ocl) and (baselines[4][0] >= b_rdi)):  
        num1 = num1+1
        basic_features.append([rsi_mean, rsi_max, b_std, b_ocl, b_rdi])
        labels.append(0)


basic_features = np.asarray(basic_features)
basic_features[:,0] = basic_features[:,0] / baselines[0][4]
basic_features[:,1] = basic_features[:,1] / baselines[1][4]
basic_features[:,2] = basic_features[:,2] / baselines[2][4]
basic_features[:,3] = basic_features[:,3] / baselines[3][4]
basic_features[:,4] = basic_features[:,4] / baselines[4][4]

np.save(os.path.join(save_dir, 'basic.npy'), basic_features)   
np.save(os.path.join(save_dir, 'labels.npy'), labels)   
     
print(basic_features)     
print(f'total: {total}')
print(num1, num2, num3, num4, num5)
print(f'rate: {(num1+num2+num3+num4+num5)/total}')



    

  


