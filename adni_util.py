import math
import os
import random
import numpy as np
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage
import pandas as pd
from deepbrain import Extractor
from skimage import measure
import itertools
from PIL import Image, ImageEnhance
from skimage.util import random_noise
import nibabel as nib
from sklearn.feature_extraction.image import extract_patches

def pixel_probability(img):
    """
    计算像素值出现概率
    :param img:
    :return:
    """
    assert isinstance(img, np.ndarray)

    prob = np.zeros(shape=(256))

    for rv in img:
        for cv in rv:
            prob[cv] += 1

    r, c = img.shape
    prob = prob / (r * c)

    return prob


def probability_to_histogram(img, prob):
    """
    根据像素概率将原始图像直方图均衡化
    :param img:
    :param prob:
    :return: 直方图均衡化后的图像
    """
    prob = np.cumsum(prob)  # 累计概率

    img_map = [int(i * prob[i]) for i in range(256)]  # 像素值映射

    # 像素值替换
    assert isinstance(img, np.ndarray)
    r, c = img.shape
    for ri in range(r):
        for ci in range(c):
            img[ri, ci] = img_map[img[ri, ci]]

    return img

def all_np(arr):
    # 拼接数组函数
    List = list(itertools.chain.from_iterable(arr))
    arr = np.array(List)
    key = np.unique(arr)
    result = {}
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        result[k] = v
    return result


def softmax(x, axis=1):
    x_sum = np.sum(x, axis=axis, keepdims=True)
    s = x / x_sum
    return s

class AdniDataSet(Dataset):

    def __init__(self, image_path, train_sub_idx, train_labels_mmse, train_labels_adas13, train_labels_adas11,
                 train_labels_cdrsb,  image_size, BIN_SIZE_MMSE, BIN_NUM_MMSE, BIN_SIZE_ADAS,
                 BIN_NUM_ADAS,  BIN_SIZE_CDRSN, BIN_NUM_CDRSN):

        self.image_path = image_path
        self.train_sub_idx = train_sub_idx
        self.train_labels_mmse = train_labels_mmse
        self.train_labels_adas13 = train_labels_adas13
        self.train_labels_adas11 = train_labels_adas11
        self.train_labels_cdrsb = train_labels_cdrsb

        self.input_D = image_size
        self.input_H = image_size
        self.input_W = image_size
        self.BIN_SIZE_MMSE = BIN_SIZE_MMSE
        self.BIN_NUM_MMSE = BIN_NUM_MMSE
        self.BIN_SIZE_MMSE = BIN_SIZE_MMSE
        self.BIN_SIZE_ADAS = BIN_SIZE_ADAS
        self.BIN_NUM_ADAS = BIN_NUM_ADAS
        self.BIN_SIZE_CDRSN = BIN_SIZE_CDRSN
        self.BIN_NUM_CDRSN = BIN_NUM_CDRSN

    def __nii2tensorarray__(self, data):
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, x, y, z])
        new_data = new_data.astype("float32")

        return new_data

    def __len__(self):
        return len(self.train_labels_mmse)

    def __getitem__(self, idx):

        image_subject = nib.load(self.image_path + self.train_sub_idx[idx] + ".nii")

        output_mmse = self.train_labels_mmse[idx]
        output_adas13 = self.train_labels_adas13[idx]
        output_adas11 = self.train_labels_adas11[idx]
        output_cdrsb = self.train_labels_cdrsb[idx]

        img_array1 = self.__training_data_process__(image_subject)
        img_array1 = self.__nii2tensorarray__(img_array1)


        '''mmse'''
        k_mmse = self.BIN_SIZE_MMSE

        z1_mmse = int(output_mmse / k_mmse)
        z2_mmse = int(output_mmse / k_mmse) + 1
        label1_mmse = 1 - (output_mmse - int(output_mmse / k_mmse) * k_mmse) / k_mmse
        label2_mmse = 1 - ((int(output_mmse / k_mmse) + 1) * k_mmse - output_mmse) / k_mmse

        distribution_mmse = np.zeros(self.BIN_NUM_MMSE)
        distribution_mmse[z1_mmse] = label1_mmse
        distribution_mmse[z2_mmse] = label2_mmse

        branch_mmse = np.zeros(self.BIN_NUM_MMSE)
        branch_mmse[:z2_mmse] = 1

        '''adas11'''
        k_adas11 = self.BIN_SIZE_ADAS

        z1_adas11 = int(output_adas11 / k_adas11)
        z2_adas11 = int(output_adas11 / k_adas11) + 1
        label1_adas11 = 1 - (output_adas11 - int(output_adas11 / k_adas11) * k_adas11) / k_adas11
        label2_adas11 = 1 - ((int(output_adas11 / k_adas11) + 1) * k_adas11 - output_adas11) / k_adas11

        distribution_adas11 = np.zeros(self.BIN_NUM_ADAS)
        distribution_adas11[z1_adas11] = label1_adas11
        distribution_adas11[z2_adas11] = label2_adas11

        branch_adas11 = np.zeros(self.BIN_NUM_ADAS)
        branch_adas11[:z2_adas11] = 1

        '''adas13'''
        k_adas13 = self.BIN_SIZE_ADAS

        z1_adas13 = int(output_adas13 / k_adas13)
        z2_adas13 = int(output_adas13 / k_adas13) + 1
        label1_adas13 = 1 - (output_adas13 - int(output_adas13 / k_adas13) * k_adas13) / k_adas13
        label2_adas13 = 1 - ((int(output_adas13 / k_adas13) + 1) * k_adas13 - output_adas13) / k_adas13

        distribution_adas13 = np.zeros(self.BIN_NUM_ADAS)
        distribution_adas13[z1_adas13] = label1_adas13
        distribution_adas13[z2_adas13] = label2_adas13

        branch_adas13 = np.zeros(self.BIN_NUM_ADAS)
        branch_adas13[:z2_adas13] = 1

        ''' cdrsb '''
        k_cdrsb = self.BIN_SIZE_MMSE

        z1_cdrsb = int(output_cdrsb / k_cdrsb)
        z2_cdrsb = int(output_cdrsb / k_cdrsb) + 1
        label1_cdrsb = 1 - (output_cdrsb - int(output_cdrsb / k_cdrsb) * k_cdrsb) / k_cdrsb
        label2_cdrsb = 1 - ((int(output_cdrsb / k_cdrsb) + 1) * k_cdrsb - output_cdrsb) / k_cdrsb

        distribution_cdrsb = np.zeros(self.BIN_NUM_CDRSN)
        distribution_cdrsb[z1_cdrsb] = label1_cdrsb
        distribution_cdrsb[z2_cdrsb] = label2_cdrsb

        branch_cdrsb = np.zeros(self.BIN_NUM_CDRSN)
        branch_cdrsb[:z2_cdrsb] = 1

        return img_array1, output_mmse, output_adas13, output_adas11, output_cdrsb, \
                           distribution_mmse, distribution_adas13, distribution_adas11, distribution_cdrsb,\
                           branch_mmse, branch_adas13, branch_adas11, branch_cdrsb

    def __itensity_normalize_one_volume__(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """

        pixels = volume[volume > 0]
        mean = pixels.mean()
        std = pixels.std()
        out = (volume - mean) / std
        out_random = np.random.normal(0, 1, size=volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out

    def __scaler__(self, image):
        img_f = image.flatten()
        # find the range of the pixel values
        i_range = img_f[np.argmax(img_f)] - img_f[np.argmin(img_f)]
        # clear the minus pixel values in images
        image = image - img_f[np.argmin(img_f)]
        img_normalized = np.float32(image / i_range)

        return img_normalized

    def __resize_data__(self, data):
        [depth, height, width] = data.shape
        scale = [self.input_D * 1.0 / depth, self.input_H * 1.0 / height, self.input_W * 1.0 / width]
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data

    def __crop_data__(self, data):
        # random center crop
        data = self.__random_center_crop__(data)

        return data

    def __training_data_process__(self, data):
        # crop data according net input size

        data = data.get_data()
        data = self.__drop_invalid_range__(data)
        # resize data
        data = self.__resize_data__(data)


        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        # data = self.__scaler__(data)


        return data



    def __drop_invalid_range__(self, volume, label=None):
        """
        Cut off the invalid area
        """
        zero_value = volume[0, 0, 0]
        non_zeros_idx = np.where(volume != zero_value)

        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)

        if label is not None:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w], label[min_z:max_z, min_h:max_h, min_w:max_w]
        else:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w]


