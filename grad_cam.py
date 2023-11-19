import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import csv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import pandas as pd
from train_cnn import CustomCNN
import cv2


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # モデルの最後の畳み込み層を取得
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # クラスの予測と特徴マップの勾配を取得
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 勾配の平均を特徴マップに掛ける
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ヒートマップを正規化
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def display_gradcam(img, heatmap):
    # 無効な値（NaNなど）をチェックし、0に置き換える
    #heatmap = np.nan_to_num(heatmap)

    # ヒートマップをRGBに変換
    heatmap = np.uint8(255 * heatmap)

    # カラーマップを使用してヒートマップをカラー画像に変換
    #jet = cm.get_cmap("jet")
    jet = mpl.colormaps['jet']

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # ジェットカラーマップを元の画像サイズにリサイズ
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[2], img.shape[1]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # ヒートマップの透明度を設定して元の画像に重ねる
    superimposed_img = jet_heatmap
    #superimposed_img = jet_heatmap * alpha + img[0]
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # 画像を表示
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.show()

def main():


    SIZE = 512
    OUTPUT_DIR = "/home/jink4869/kiyota/model"

    custom_cnn = CustomCNN(size=SIZE, output_dir=OUTPUT_DIR)
    loaded_model = CustomCNN.load_model(os.path.join(OUTPUT_DIR,'model.h5'))

    # 各画像の推論
    x_val = np.load("/home/jink4869/kiyota/stage/test_after_aug/train_image.npy")
    for i in range(6):
        predictions = loaded_model.predict(x_val[i*100:(i+1)*100])
        print(np.average(np.dot(predictions,[0,1,2,3,4,5])))


    # Grad-CAMを実行

    img_array = cv2.imread("/home/jink4869/kiyota/stage/test_after_aug/0/00000000.png")/255
    img_array = img_array[np.newaxis,...]

    last_conv_layer_name = 'conv2d_5'
    heatmap = make_gradcam_heatmap(img_array, loaded_model, last_conv_layer_name)
    display_gradcam(img_array, heatmap)






if __name__ == '__main__':
    from tensorflow.keras import mixed_precision

    # mixed_precisionを'mixed_float16'に設定
    mixed_precision.set_global_policy('mixed_float16')

    main()






