import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import csv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import pandas as pd

class CustomCNN:
    def __init__(self, size,output_dir):

        self.output_dir = output_dir
        self.build(size)
        self.initialize()


    def initialize(self):
        os.makedirs(self.output_dir, exist_ok=True)


    def build(self, size):
        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(size, size, 3), name='conv2d_1'),
            layers.MaxPooling2D((2, 2), name='max_pooling2d_1'),
            layers.Conv2D(32, (5, 5), activation='relu', name='conv2d_2'),
            layers.MaxPooling2D((2, 2), name='max_pooling2d_2'),
            layers.Conv2D(32, (5, 5), activation='relu', name='conv2d_3'),
            layers.MaxPooling2D((2, 2), name='max_pooling2d_3'),
            layers.Conv2D(32, (5, 5), activation='relu', name='conv2d_4'),
            layers.MaxPooling2D((2, 2), name='max_pooling2d_4'),
            layers.Conv2D(32, (3, 3), activation='relu', name='conv2d_5'),
            layers.MaxPooling2D((2, 2), name='max_pooling2d_5'),
            layers.Flatten(name='flatten'),
            layers.Dense(32, activation='relu', name='dense_1'),
            layers.Dense(6, activation='softmax', name='output')
        ])

        self.model.compile(optimizer=optimizers.Adam(learning_rate=0.00001),
                            loss="sparse_categorical_crossentropy",
                            metrics=['accuracy'])
        self.model.summary()


    def repeat_fit(self, x_train, y_train, x_val, y_val):

        with open(os.path.join(self.output_dir,'training_history.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            # CSVのヘッダーを書き込む
            writer.writerow(['Repeat', 'Epoch', 'Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy'])

        batch_size = 1  # Define your batch size
        SIZE = 512
        EPOCHS = 2
        DATA_SIZE = 600
        REPEAT = 10

        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        for i in range (REPEAT):
            # Create TensorFlow datasets
            random_indices_train = np.random.choice(x_train.shape[0], DATA_SIZE, replace=False)
            train_dataset = tf.data.Dataset.from_tensor_slices((x_train[random_indices_train], y_train[random_indices_train])).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

            # Now use these datasets in your model's fit method
            history = self.model.fit(
                train_dataset,
                epochs=EPOCHS,
                validation_data=val_dataset
            )

            with open(os.path.join(self.output_dir,'training_history.csv'), 'a', newline='') as file:
                writer = csv.writer(file)
                for epoch in range(EPOCHS):
                    writer.writerow([
                    i,
                    epoch,
                    history.history['loss'][epoch],
                    history.history['accuracy'][epoch],
                    history.history['val_loss'][epoch],
                    history.history['val_accuracy'][epoch]
                    ])

        self.save_model(os.path.join(self.output_dir,"model.h5"))
        self.plot_training_history(os.path.join(self.output_dir,'training_history.csv'))


    def plot_training_history(self, csv_file):
        # CSVファイルを読み込む
        df = pd.read_csv(csv_file)

        # データの数（行数）を取得
        data_points = range(len(df))

        # 精度（Accuracy）のグラフを描画
        plt.figure()
        plt.plot(data_points, df['Train Accuracy'], label='Training Accuracy')
        plt.plot(data_points, df['Validation Accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Data Point')
        plt.ylabel('Accuracy')
        plt.savefig(os.path.join(self.output_dir,"acc.png"))

        # 損失（Loss）のグラフを描画
        plt.cla()
        plt.plot(data_points, df['Train Loss'], label='Training Loss')
        plt.plot(data_points, df['Validation Loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Data Point')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir,"loss.png"))


    def save_model(self, file_path):
        self.model.save(file_path)

    @staticmethod
    def load_model(file_path):
        return tf.keras.models.load_model(file_path, compile=False)


def main():


    x_train = np.load("/home/jink4869/kiyota/stage/train_after_aug/train_image.npy")
    y_train = np.load("/home/jink4869/kiyota/stage/train_after_aug/train_label.npy")

    x_val = np.load("/home/jink4869/kiyota/stage/test_after_aug/train_image.npy")
    y_val = np.load("/home/jink4869/kiyota/stage/test_after_aug/train_label.npy")

    SIZE = 512
    OUTPUT_DIR = "/home/jink4869/kiyota/model"

    custom_cnn = CustomCNN(size=SIZE, output_dir=OUTPUT_DIR)
    custom_cnn.repeat_fit(x_train, y_train, x_val, y_val)

    # loaded_model = CustomCNN.load_model('my_model.h5')


if __name__ == '__main__':
    from tensorflow.keras import mixed_precision

    # mixed_precisionを'mixed_float16'に設定
    mixed_precision.set_global_policy('mixed_float16')

    main()






