from __future__ import print_function
import time
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import tensorflow as tf
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
from scipy import misc
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import cv2, os


ImageFile.LOAD_TRUNCATED_IMAGES = True

PATH = os.path.join(os.path.dirname('/Users/kevin/Documents/ML Related/Dataset/'), 'KDEF')

os.chdir('/Users/kevin/Documents/ML Related/Dataset/')
#image = cv2.imread()
arr = os.listdir(PATH)

target_KDEF = np.array([])
data_KDEF = np.array([])
target_MMISEL = np.array([])
data_MMISEL = np.array([])
for k in range(len(arr)):
    arr1 = os.listdir(PATH + '/' + arr[k])
    for j in range(len(arr1)):
        image = Image.open(PATH + '/' + arr[k] + '/' + arr1[j])
        image = np.array(image.resize((48,60)))
        for i in range(3):
            rgb = image[:,:,i]
            # rgb = rgb[np.newaxis,:]
            rgb = rgb.flatten()
            print(rgb.shape)
            if data_KDEF.size == 0:
                data_KDEF = rgb
            else:
                data_KDEF = np.vstack((data_KDEF,rgb))
                print(data_KDEF.shape)
            if i == 0:
                target_KDEF = np.append(target_KDEF,('KDEF_R'))
            elif i == 1:
                target_KDEF = np.append(target_KDEF,('KDEF_G'))
            elif i == 2:
                target_KDEF = np.append(target_KDEF,('KDEF_B'))
print(data_KDEF.shape, target_KDEF.shape)

PATH = os.path.join(os.path.dirname('/Users/kevin/Documents/ML Related/Dataset/'), 'MMI_selected')
arr = os.listdir(PATH)

for k in range(len(arr)):
    arr1 = os.listdir(PATH + '/' + arr[k])
    for j in range(len(arr1)):
        image = Image.open(PATH + '/' + arr[k] + '/' + arr1[j])
        image = np.array(image.resize((48,60)))
        for i in range(3):
            rgb = image[:,:,i]
            rgb = rgb.flatten()
            # rgb = rgb[np.newaxis,:]
            if data_MMISEL.size == 0:
                data_MMISEL = rgb
            else:
                data_MMISEL = np.vstack((data_MMISEL, rgb))
            if i == 0:
                target_MMISEL = np.append(target_MMISEL,('MMISEL_R'))
            elif i == 1:
                target_MMISEL = np.append(target_MMISEL,('MMISEL_G'))
            elif i == 2:
                target_MMISEL = np.append(target_MMISEL,('MMISEL_B'))

print(data_MMISEL.shape)
print(target_MMISEL.shape)
print(data_KDEF.shape)
print(target_KDEF.shape)

# np.save('tsne_data_RGB_KDEF_.npy', data_KDEF)
# np.save('tsne_target_RGB_KDEF_.npy', target_KDEF)
# np.save('tsne_data_RGB_MMISEL_.npy', data_MMISEL)
# np.save('tsne_target_RGB_MMISEL_.npy', target_MMISEL)

f= np.load('/Users/kevin/Documents/ML Related/Dataset/tsne_data_RGB_KDEF.npy')

print(f.shape)

# fig = plt.figure(figsize=(4,4))

# for i in range(4):
#     plt.subplot(4, 4, i+1)
#     plt.imshow(f[i, :, :] )
#     plt.axis('off')

# plt.savefig('/content/drive/My Drive/Colab Notebooks/images/MMISEL_GB_at_epoch_{:04d}.png'.format(epoch))
plt.show()