import datetime
import numpy as np
import tensorflow as tf
from network import GoogLeNet

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu,True)

class DataLoader():
    def __init__(self):
        initial_data = tf.keras.datasets.fashion_mnist
        (self.train_images, self.train_labels),(self.test_images, self.test_labels) = initial_data.load_data()
        # 将图片转为float32且除以255进行归一化; expand_dims增加维度
        self.train_images = np.expand_dims(self.train_images.astype(np.float32) / 255.0, axis=-1)
        self.test_images = np.expand_dims(self.test_images.astype(np.float32) / 255.0, axis=-1)
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

def train_googlenet(batch_size, epoch):
    #　加载数据
    dataLoader = DataLoader()
    train_images, train_labels = dataLoader.get_batch_train(60000)
    # 创建和编译模型
    model = GoogLeNet.build_goolenet()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=False)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # 创建检查点,用于保存权重;创建的checkpoint在fit()时作为回调函数传递
    checkpoint = tf.keras.callbacks.ModelCheckpoint(f'./weight/{epoch}_epoch_goolenet_weight.h5', save_weights_only=True, save_best_only=True, verbose=1, save_freq='epoch')
    # 创建tensorboard,在fit()时回调
    log_dir = "resource/log/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
    # train
    model.fit(train_images, train_labels, epochs=epoch, shuffle=True, batch_size=batch_size, validation_split=0.1, callbacks=[checkpoint,tensorboard])

def test_googlenet(weight_path, batch_size):
    dataLoader = DataLoader()
    test_images, test_labels = dataLoader.get_batch_test(batch_size)
    model = GoogLeNet.build_goolenet()
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.build((1, 64, 64, 1))
    model.load_weights(weight_path)
    model.evaluate(test_images, test_labels)

if __name__ == '__main__':
    # 训练
    # train_googlenet(256,20)

    # 测试
    test_googlenet('./weight/20_epoch_goolenet_weight.h5', 10000) # acc 90.4%