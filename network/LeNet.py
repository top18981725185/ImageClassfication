import tensorflow as tf
from tensorflow import keras

class LeNetModel(tf.keras.Model):
    """"构建LeNet模型---通过自定义子类化Model实现"""
    def __init__(self):
        super(LeNetModel,self).__init__()
        self.conv1 = keras.layers.Conv2D(filters=6, kernel_size=5, activation='relu', input_shape=(28,28,1))
        self.pool = keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.conv2 = keras.layers.Conv2D(filters=16, kernel_size=5, activation='relu')
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(120, activation='relu')
        self.dense2 = keras.layers.Dense(84, activation='relu')
        self.dense3 = keras.layers.Dense(10, activation='softmax')
        self.dropout = keras.layers.Dropout(0.25)

    def call(self,inputs,traing=False):
        x = self.dense1(self.flatten(self.pool(self.conv2(self.pool(self.conv1(inputs))))))
        if traing:
            x = self.dropout(self.dense2(self.dropout(x,traing=traing)))
        else:
            x = self.dense2(x)
        return self.dense3(x)

def build_suquential_model():
    """"构建LeNet模型---通过Suquential()实现"""
    net = keras.models.Sequential([
    keras.layers.Conv2D(filters=6, kernel_size=5, activation='relu',input_shape=(28,28,1)),
    keras.layers.MaxPool2D(pool_size=2, strides=2),
    keras.layers.Conv2D(filters=16, kernel_size=5, activation='relu'),
    keras.layers.MaxPool2D(pool_size=2,strides=2),
    keras.layers.Flatten(),
    keras.layers.Dense(120, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(84, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(10, activation='softmax')
    ])
    return net

def build_functional_model():
    """"构建LeNet模型---functional API实现"""
    inputs = keras.layers.Input([28,28,1])
    conv1 = keras.layers.Conv2D(filters=6, kernel_size=5, activation='relu', input_shape=(28,28,1))(inputs)
    pool1 = keras.layers.MaxPool2D(pool_size=2, strides=2)(conv1)
    conv2 = keras.layers.Conv2D(filters=16, kernel_size=5, activation='relu')(pool1)
    pool2 = keras.layers.MaxPool2D(pool_size=2, strides=2)(conv2)
    flatten = keras.layers.Flatten()(pool2)
    dense1 = keras.layers.Dense(120,activation='relu')(flatten)
    dropout1 = keras.layers.Dropout(0.25)(dense1)
    dense2 = keras.layers.Dense(84,activation='relu')(dropout1)
    dropout2 = keras.layers.Dropout(0.25)(dense2)
    dense3 = keras.layers.Dense(10,activation=None)(dropout2)
    outputs = tf.nn.softmax(dense3)
    net = keras.Model(inputs=inputs,outputs=outputs)
    return net

def build_lenet(keyword='sequential'):
    if keyword=='sequential':
        return build_suquential_model()
    if keyword=='functional':
        return build_functional_model()
    if keyword=='subclass':
        return LeNetModel()



