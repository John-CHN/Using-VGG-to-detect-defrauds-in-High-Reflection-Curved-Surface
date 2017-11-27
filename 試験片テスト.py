# coding:utf-8
import keras
import sys, os
import scipy
import scipy.misc
import numpy as np******
from keras.models import model_from_json
import h5py
import json
#JSON(JavaScript Object Notation) 是一种轻量级的数据交换格式。易于人阅读和编写。同时也易于机器解析和生成。
#Python的Json模块序列化与反序列化的过程分别是 encoding和 decoding。使用json模块必须先导入
imsize = (64, 64)

# 指定要识别的图像路径
# ./blog_testpic/xxx.jpg といった指定を意味する
testpic = "C:/Users/Borui/Desktop/dataset/testset/"
# 指定要使用的模型
keras_model = "C:/Users/Borui/Desktop/dataset/idol.json"

keras_param = "C:/Users/Borui/Desktop/dataset/idol-model.hdf5"


# 加载图像
# 使导入的图像 X
def load_image(path):
    img = scipy.misc.imread(path, mode="RGB")
    img = scipy.misc.imresize(img, imsize)
    img = img / 255.0  # RGBの最大値を指定
                       # 指定RGB的最大值
    return img


# リストで結果を返す関数
# 函数返回列表中的结果
def get_file(dir_path):
    filenames = os.listdir(dir_path) #列出dirname下的目录和文件
    return filenames


# main開始
if __name__ == "__main__":

    # 阅读图像并列出文件名称。
    pic = get_file(testpic)
    print(pic)


    # 加载模型
    model = model_from_json(open(keras_model).read())
    model.load_weights(keras_param)
    model.summary()
    #到目前为止，模型形状将显示在结果中

    # 从列出的文件中读取并处理它们
    for i in pic:
        print(i)  # 输出文件名称

        # 读取图像目录中的文件的第i个
        img = load_image(testpic + i)

        prd = model.predict(np.array([img]))
        print(prd)

        # 获得最大的置信度
        prelabel = np.argmax(prd, axis=1)
        print(prelabel)

        # 每个图像文件中的飞机为0，狗为5，船为8
        if prelabel == 0:
            print(">>> 試験片あり")
        elif prelabel == 1:
            print(">>> 試験片なし")

        print()
        print()