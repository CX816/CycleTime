# Acknowledgments
Our code is inspired by [pytorch-DCGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
from pyts.image import GramianAngularField
import matplotlib.pyplot as plt

# 使用GADF将一维时序数据生成二维数据并保存
gadf_transformer = GramianAngularField(method='difference')
for i, (label, feature) in enumerate(zip(labels, features)):
    gadf_image = gadf_transformer.transform(feature.reshape(1, -1))
    filename = f'trainRefrigerationDevices/image_{i}_label{label}.png'
    plt.imsave(filename, gadf_image[0], cmap='rainbow')
