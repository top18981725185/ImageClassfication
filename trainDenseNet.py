import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from network import DenseNet

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu,True)

class DataLoader():
    def __init__(self):
        initial_data = tf.keras.datasets.cifar10
        (self.train_images, self.train_labels),(self.test_images, self.test_labels) = initial_data.load_data()
        # 将图片转为float32且除以255进行归一化;
        self.train_images = self.train_images.astype(np.float32) / 255.0
        self.test_images = self.test_images.astype(np.float32) / 255.0
        self.train_labels = self.train_labels.astype(np.int32)
        self.test_labels = self.test_labels.astype(np.int32)
        self.num_train, self.num_test = self.train_images.shape[0], self.test_images.shape[0]

    def get_batch_train(self, batch_size):
        # np.random.randint均匀分布,从训练集中随机产生batch_size个索引
        index = np.random.randint(0, np.shape(self.train_images)[0], batch_size)
        # 将图片resize至合适大小,这里使用224*224,使用32*32会报维度错误
        resized_images = tf.image.resize_with_pad(self.train_images[index], 64, 64,)
        return  resized_images.numpy(),self.train_labels[index]

    def get_batch_test(self, batch_size):
        index = np.random.randint(0, np.shape(self.test_images)[0], batch_size)
        resized_images = tf.image.resize_with_pad(self.test_images[index], 64, 64,)
        return resized_images.numpy(), self.test_labels[index]

def train_densenet(batch_size, epoch):
    dataLoader = DataLoader()
    # build callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(f'./weight/{epoch}_epoch_densenet_weight.h5', save_best_only=True, save_weights_only=True, verbose=1, save_freq='epoch')
    # build model
    net = DenseNet.build_densenet()
    net.compile(tf.keras.optimizers.Adam(lr=0.001, decay=1e-6), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # num_iter = dataLoader.num_train//batch_size
    # for e in range(epoch):
    #     for i in range(num_iter):
    #         train_images, train_labels = dataLoader.get_batch_train(batch_size)
    #         net.fit(train_images, train_labels, shuffle=False, batch_size=batch_size, validation_split=0.1, callbacks=[checkpoint])
    #     net.save_weights("./weight/"+str(e+1)+"epoch_iter"+str(i)+"_resnet_weight.h5")

    data_generate = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_epsilon=1e-6,
        zca_whitening=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.,
        zoom_range=0.1,
        channel_shift_range=0,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format='channels_last',
        validation_split=0.0)

    train_images, train_labels = dataLoader.get_batch_train(60000)
    net.fit(data_generate.flow(train_images, train_labels, batch_size=batch_size, shuffle=True,),
            steps_per_epoch=len(train_images)//batch_size,
            epochs=epoch,
            callbacks=[checkpoint],
            shuffle=True)

def test_densenet(modelpath, batch_size):
    dataLoader = DataLoader()
    net = DenseNet.build_densenet()
    net.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    net.build((1,64,64,3))
    net.load_weights(modelpath)
    test_images, test_labels = dataLoader.get_batch_test(batch_size)
    net.evaluate(test_images, test_labels, verbose=2)

if __name__ == '__main__':
    # 训练
    # train_densenet(256, 60)
    test_densenet('./weight/')