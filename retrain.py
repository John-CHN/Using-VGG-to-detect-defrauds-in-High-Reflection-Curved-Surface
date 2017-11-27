from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras.models import load_model
from keras.models import model_from_json
import numpy as np
from keras import initializers
model = Sequential()
categories = ["試験片あり","試験片なし"]
nb_classes = len(categories)
in_shape=(64,64,3)
#VGG 传统卷积神经网络
# 创建卷积层
# 首先，添加第一层的1024层，并创建16个过滤器3 * 3的过滤器
model.add(Convolution2D(16, (3, 3), border_mode="same",
                        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1, seed=None),activation="relu",input_shape=in_shape)),

# 第二层的卷积层
model.add(Convolution2D(16, (3, 3), border_mode="same",activation="relu"))


# プーリング層  合并图层
model.add(MaxPooling2D(pool_size=(2,2)))

# Dropoutとは過学習を防ぐためのもの　0.5は次のニューロンへのパスをランダムに半分にするという意味
# Dropout是为了防止过度学习,0.5意味着随机将路径减半到下一个神经元
model.add(Dropout(0.5))
#为输入数据施加Dropout。Dropout将在训练过程中每次更新参数时按一定概率（rate）随机断开输入神经元
# Dropout层用于防止过拟合
# ３層目の作成

model.add(Convolution2D(32, (3, 3), border_mode="same", activation="relu"))
# ４層目の作成
model.add(Convolution2D(32, (3, 3), border_mode="same", activation="relu"))
# プーリング層合并图层
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

# ５層目
model.add(Convolution2D(64, (3, 3), border_mode="same", activation="relu"))

# 6層目
model.add(Convolution2D(64, (3, 3), border_mode="same", activation="relu"))

# プーリング層合并图层
model.add(MaxPooling2D(pool_size=(2,2)))

# Dropout
model.add(Dropout(0.5))

# 7層目
model.add(Convolution2D(128, (3, 3), border_mode="same", activation="relu"))


# Dropout
model.add(Dropout(0.5))

# 平坦化
model.add(Flatten())

# 8層目　全結合層　FC
model.add(Dense(100))
model.add(Activation("relu"))

# Dropout
model.add(Dropout(0.5))

# 8層目　引数nub_classesとは分類の数を定義する。第八层参数nub_classes定义了分类的数量。
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# ここまででモデルの層完成   到目前为止完成了模型层

# lossは損失関数を定義するところ 损失定义了损失函数
model.compile(loss="categorical_crossentropy",
              metrics=["accuracy"],
              optimizer="adam"
              )
X_train, X_test, y_train, y_test = np.load("C:/Users/Borui/Desktop/Predata/cutidol.npy")
X_train = X_train.astype("float") / 256
X_test  = X_test.astype("float")  / 256
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test  = np_utils.to_categorical(y_test, nb_classes)
hist=model.fit(X_train, y_train, epochs=1, batch_size=128,validation_split=0.7)

#fit函数返回一个History的对象，其History.history属性记录了损失函数和其他指标的数值随epoch变化的情况，
# 如果有验证集的话，也包含了验证集的这些指标变化情况
print(hist.history)
import matplotlib.pyplot as plt
# list all data in history
print(hist.history.keys())
# summarize history for accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()