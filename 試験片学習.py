from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import numpy as np
import h5py
#类别进行分类"./photo_out/
root_dir = "C:/Users/Borui/Desktop/Predata"
categories = ["試験片あり","試験片なし"]
nb_classes = len(categories)
#image_size = 50
 
# -データをロード -- (※1)
#载入数据---（* 1）
def main():
    X_train, X_test, y_train, y_test = np.load("C:/Users/Borui/Desktop/Predata/cutidol.npy")
    # データを正規化する
    #规范化数据
    X_train = X_train.astype("float") / 256
    X_test  = X_test.astype("float")  / 256
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test  = np_utils.to_categorical(y_test, nb_classes)
    # モデルを訓練し評価する  \培训和评估模型
    model = model_train(X_train, y_train)
    model_eval(model, X_test, y_test)
 
# モデルを構築 --- (※2)
def build_model(in_shape):
    model = Sequential()
    
    print(in_shape)
    #创建卷积层
    #首先，添加第一层的1024层，并创建16个过滤器3 * 3的过滤器
    model.add(Convolution2D(16, 3, 3, border_mode="same", input_shape=in_shape))
    model.add(Activation("relu"))
    
    #第二层的卷积层
    model.add(Convolution2D(16, 3, 3, border_mode="same"))
    model.add(Activation("relu"))
    
     #プーリング層  合并图层
    model.add(MaxPooling2D())
    
    #Dropoutとは過学習を防ぐためのもの　0.5は次のニューロンへのパスをランダムに半分にするという意味
    #Dropout是为了防止过度学习,0.5意味着随机将路径减半到下一个神经元
    model.add(Dropout(0.5))
    
    #３層目の作成
    model.add(Convolution2D(32, 3, 3, border_mode="same"))
    model.add(Activation("relu"))
    
    #４層目の作成
    model.add(Convolution2D(32, 3, 3, border_mode="same"))
    model.add(Activation("relu"))
    
     #プーリング層合并图层
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))
    
    #５層目
    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(Activation("relu"))
    
    #6層目
    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(Activation("relu"))
    
    #プーリング層合并图层
    model.add(MaxPooling2D())
    
    #Dropout
    model.add(Dropout(0.5))
    
    #7層目
    model.add(Convolution2D(128, 3, 3, border_mode="same"))
    model.add(Activation("relu"))
    
    #Dropout
    model.add(Dropout(0.5))
    
    #平坦化
    model.add(Flatten())
    
    #8層目　全結合層　FC
    model.add(Dense(100))
    model.add(Activation("relu"))
    
    #Dropout
    model.add(Dropout(0.5))
    
    #8層目　引数nub_classesとは分類の数を定義する。第八层参数nub_classes定义了分类的数量。
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    #ここまででモデルの層完成   到目前为止完成了模型层
    
    #lossは損失関数を定義するところ 损失定义了损失函数
    model.compile(loss="categorical_crossentropy", 
        metrics   = ["accuracy"], 
        optimizer = "adam"
    )
    return model
 
 
# モデルを訓練する 训练模型--- (※3)训练模型来训练一个模型
def model_train(X, y):
    model = build_model(X.shape[1:])
    model.fit(X, y,  batch_size=50, nb_epoch=3,validation_split=0.1)
    hist = model.fit(X, y, validation_split=0.1)
    print(hist.history)
    
    #学習モデルの保存 保存学习model
    json_string = model.to_json()
     #モデルのファイル名　拡張子.json
    open('C:/Users/Borui/Desktop/Predata/cutidol.json', 'w').write(json_string)
    
    # model を保存する --- (※4)
    hdf5_file = "C:/Users/Borui/Desktop/Predata/cutidol-model.hdf5"
    model.save_weights(hdf5_file)
    return model
 
# model を評価する --- (※5)
def model_eval(model, X, y):
    score = model.evaluate(X, y)
    print('loss=', score[0])
    print('accuracy=', score[1])
 
if __name__ == "__main__":
    main()