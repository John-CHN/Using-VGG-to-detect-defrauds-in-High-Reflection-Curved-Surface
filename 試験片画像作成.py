from PIL import Image
import os, glob
import numpy as np
import random, math

# 指定图像存储的目录路径
root_dir = "C:/Users/Borui/Desktop/Predata"
# 请指定图像按顺序排序的文件夹名称。
categories = ["試験片あり", "試験片なし"]
image_size = 50

# 读取图像数据
X = []  # 图像数据
Y = []  # 标签数据


def add_sample(cat, fname, is_train):
    img = Image.open(fname)
    img = img.convert("RGB")  # 更改颜色模式
    img = img.resize((image_size, image_size))  # 改变图像大小
    data = np.asarray(img)
    X.append(data)
    Y.append(cat)  # cat is the number0/1, good is 1 ,defects is 0,   X is the data
    if not is_train: return
    # 添加不同角度的数据
    # 一点一点旋转
    for ang in range(-20, 20, 5):
        img2 = img.rotate(ang)
        data = np.asarray(img2)
        X.append(data)
        Y.append(cat)  # append() 方法向列表的尾部添加一个新的元素。只接受一个参数。
        # 反転する
        img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
        data = np.asarray(img2)
        X.append(data)
        Y.append(cat)


def make_sample(files, is_train):
    global X, Y
    X = [];
    Y = []
    for cat, fname in files:
        add_sample(cat, fname, is_train)
    return np.array(X), np.array(Y)


# 收集为每个目录分割的文件
allfiles = []
for idx, cat in enumerate(categories):
    image_dir = root_dir + "/" + cat
    files = glob.glob(image_dir + "/*.jpg")
    for f in files:
        allfiles.append((idx, f))  # good is 1, defect is 0;

# 随机播放并分成学习数据和测试数据
random.shuffle(allfiles)
th = math.floor(len(allfiles) * 0.95)  # 0.7 means training data 占比，0.3是test data 占比
train = allfiles[0:th]
test = allfiles[th:]
X_train, y_train = make_sample(train, True)
X_test, y_test = make_sample(test, False)
xy = (X_train, X_test, y_train, y_test)
np.save("C:/Users/Borui/Desktop/Predata/cutidol.npy", xy)
print("ok,", len(y_train))
