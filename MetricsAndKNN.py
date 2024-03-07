import cv2
import os
from glob import glob
import numpy as np
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from features import get_features


########## modify the path to match your configuration.##############             
data_root = [fr'.\\sample\\images']
save_dir = f'.\\sample/afterKNN'
baseline_dir = fr'.\\baseline'

# make directory
level5 = os.path.join(save_dir,'4')
os.makedirs(level5, exist_ok=True)
level4 = os.path.join(save_dir,'3')
os.makedirs(level4, exist_ok=True)
level3 = os.path.join(save_dir,'2')
os.makedirs(level3, exist_ok=True)
level2 = os.path.join(save_dir,'1')
os.makedirs(level2, exist_ok=True)
level1 = os.path.join(save_dir,'0')
os.makedirs(level1, exist_ok=True)


#load training data .npy
X_train = np.load(os.path.join(baseline_dir, 'basic.npy'))
y_train = np.load(os.path.join(baseline_dir, 'labels.npy'))
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)



# set weight of each metrics
########
# rsi_mean
# X_train[:,0] = X_train[:,0]
# rsi_max
# X_train[:,1] = X_train[:,1]
# b_std 
X_train[:,2] = X_train[:,2]*0.8
# b_ocl
X_train[:,3] = X_train[:,3]*0.5
# b_rdi
# X_train[:,4] = X_train[:,4]

# read baseline values for normoalization
baselines = []

with open(os.path.join(baseline_dir, 'baseline.txt'), 'r') as f:
    for line in f:
        values = line.split()
        baselines.append([float(val) for val in values])

baselines = np.array(baselines)



def normalize(rsi_mean, rsi_max, b_std, b_ocl, b_rdi, baselines):
    rsi_mean = rsi_mean / baselines[0][4]
    rsi_max = rsi_max / baselines[1][4]
    b_std = b_std / baselines[2][4]
    b_ocl = b_ocl / baselines[3][4]
    b_rdi = b_rdi / baselines[4][4]
    
    return rsi_mean, rsi_max, b_std, b_ocl, b_rdi


#build KNN
KNN = KNeighborsClassifier(n_neighbors=5)
#train
KNN.fit(X_train, y_train)  


# read informations
img_path_list = []
for data_root_tmp in data_root:
    img_path_list.extend(sorted(glob(os.path.join(data_root_tmp, '**', '*.bmp'), recursive=True))) 

    
# statistic
total = len(img_path_list)
nums = {'0':0, '1':0, '2':0, '3':0, '4':0}
num_block_h = 5 # default
num_block_w = 3 # default

# loading images 
for img_path in img_path_list:
    img = cv2.imread(img_path, 0)
    
    # in windows
    img_name = img_path.split('\\')[-1]
    # in linux/MAC
    # img_name = img_path.split('/')[-1]
    
    rsi_mean, rsi_max, b_std, b_ocl, b_rdi = get_features(img, num_block_h, num_block_w)
    rsi_mean, rsi_max, b_std, b_ocl, b_rdi = normalize(rsi_mean, rsi_max, b_std, b_ocl, b_rdi, baselines)
    
    # weight of each metrics (Ensure that the weights are the same as the settings above)
    # rsi_mean = rsi_mean
    # rsi_max = rsi_max
    b_std = b_std*0.8
    b_ocl = b_ocl*0.5
    # b_rdi = b_rdi
    
    features = np.array([rsi_mean, rsi_max, b_std, b_ocl, b_rdi])
    features = np.reshape(features, (1,-1))
    pred = KNN.predict(features)[0]
    
    nums[str(pred)] += 1 
    save_path = os.path.join(save_dir, str(pred), img_name)
    cv2.imwrite(save_path, img)
    
print(f'total number of images: {total}')    
print(nums)    
    
    