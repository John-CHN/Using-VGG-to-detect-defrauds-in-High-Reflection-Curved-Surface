from keras.layers import Convolution2D, MaxPooling2D, Activation,Dropout,Flatten, Dense
from keras.models import Sequential
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pylab
#import h5py
model = Sequential()
in_shape=(320,520,3)
#print(in_shape)
# 创建卷积层
# 首先，添加第一层的1024层，并创建16个过滤器3 * 3的过滤器
model.add(Convolution2D(1, 3, 3, border_mode="same", input_shape=in_shape))
model.add(Activation("relu"))

# 第二层的卷积层
model.add(Convolution2D(1, 3, 3, border_mode="same"))
model.add(Activation("relu"))

# プーリング層  合并图层
model.add(MaxPooling2D())

# Dropoutとは過学習を防ぐためのもの　0.5は次のニューロンへのパスをランダムに半分にするという意味
# Dropout是为了防止过度学习,0.5意味着随机将路径减半到下一个神经元
model.add(Dropout(0.5))

# ３層目の作成
model.add(Convolution2D(1, 3, 3, border_mode="same"))
model.add(Activation("relu"))

# ４層目の作成
model.add(Convolution2D(1, 3, 3, border_mode="same"))
model.add(Activation("relu"))

# プーリング層合并图层
model.add(MaxPooling2D())
model.add(Dropout(0.5))

# ５層目
model.add(Convolution2D(1, 3, 3, border_mode="same"))
model.add(Activation("relu"))

# 6層目
model.add(Convolution2D(1, 3, 3, border_mode="same"))
model.add(Activation("relu"))

# プーリング層合并图层
model.add(MaxPooling2D())

# Dropout
model.add(Dropout(0.5))

# 7層目
model.add(Convolution2D(1, 3, 3, border_mode="same"))
model.add(Activation("relu"))

# Dropout
model.add(Dropout(0.5))

# 平坦化
#model.add(Flatten())

# 8層目　全結合層　FC
#model.add(Dense(100))
#model.add(Activation("relu"))

# Dropout
#model.add(Dropout(0.5))

# 8層目　引数nub_classesとは分類の数を定義する。第八层参数nub_classes定义了分类的数量。
#model.add(Dense(nb_classes))
#model.add(Activation('softmax'))

fname="C:/Users/Borui/Desktop/Predata/試験片なし/_20171018_212714_OK_org3.jpg"
cat=np.array(Image.open(fname))
print(cat.size,cat.shape)
cat_batch=np.expand_dims(cat,axis=0)
conv_cat=model.predict(cat_batch)
def visualize_cat(cat_batch):
    cat=np.squeeze(cat_batch,axis=0)
    print(cat.shape)
    plt.imshow(cat)
    pylab.show()
def nice_cat_printer(model, cat):
    '''prints the cat as a 2d array'''
    cat_batch = np.expand_dims(cat,axis=0)
    conv_cat2 = model.predict(cat_batch)

    conv_cat2 = np.squeeze(conv_cat2, axis=0)
    print( conv_cat2.shape)
    conv_cat2 = conv_cat2.reshape(conv_cat2.shape[:2])

    print (conv_cat2.shape)
    plt.imshow(conv_cat2)
    pylab.show()

nice_cat_printer(model, cat)