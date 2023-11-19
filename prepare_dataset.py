import numpy as np
import cv2
import glob
import os
from random import randint, choice


def main():



    """
    ### ステージごとの切り離し
    stage_list = [14,25,38,50,62,68]
    befor_aug = "/home/jink4869/kiyota/stage/before_aug"

    origin_path = "/home/jink4869/kiyota/deff_scale/0"
    origin_files = glob.glob(os.path.join(origin_path, "*"))
    origin_files.sort()
    for i in range(len(origin_files)):
        img = cv2.imread(origin_files[i])
        for j in range(len(stage_list)):
            if i<stage_list[j]:
                tmp = j
                break
        cv2.imwrite(os.path.join(befor_aug,str(tmp),str(i).zfill(8) + ".png"),img)



    ### データ拡張
    for i in range(6):
        befor_aug = "/home/jink4869/kiyota/stage/before_aug"
        after_aug = "/home/jink4869/kiyota/stage/after_aug"

        befor_aug_files = glob.glob(os.path.join(befor_aug,str(i),"*"))
        befor_aug_files.sort()
        for j in range(1):
        #for j in range(1,len(befor_aug_files)):
            img = cv2.imread(befor_aug_files[j])
            for k in range(100):
                tmp = random_shift_flip(img)
                cv2.imwrite(os.path.join(after_aug,str(i),str(j*100+k).zfill(8) + ".png"),tmp)



    ### .npy形式でデータ保存 train
    train_files = glob.glob(os.path.join("/home/jink4869/kiyota/stage/train_after_aug","*","*"))
    train_files.sort()

    train_image = np.zeros((len(train_files),512,512,3),np.float32)
    train_label = np.zeros((len(train_files)),np.uint8)

    for i in range(len(train_files)):
        tmp = cv2.imread(train_files[i])
        tmp = np.array(tmp,dtype=np.float32)
        tmp = tmp/255
        train_image[i] = tmp
        train_label[i] = train_files[i].split("/")[-2]

    np.save('/home/jink4869/kiyota/stage/train_after_aug/train_image.npy', train_image)
    np.save('/home/jink4869/kiyota/stage/train_after_aug/train_label.npy', train_label)



    ### .npy形式でデータ保存 test
    train_files = glob.glob(os.path.join("/home/jink4869/kiyota/stage/test_after_aug","*","*"))
    train_files.sort()
    print(len(train_files))

    train_image = np.zeros((len(train_files),512,512,3),np.float32)
    train_label = np.zeros((len(train_files)),np.uint8)

    for i in range(len(train_files)):
        tmp = cv2.imread(train_files[i])
        tmp = np.array(tmp,dtype=np.float32)
        tmp = tmp/255
        train_image[i] = tmp
        train_label[i] = train_files[i].split("/")[-2]

    np.save('/home/jink4869/kiyota/stage/test_after_aug/train_image.npy', train_image)
    np.save('/home/jink4869/kiyota/stage/test_after_aug/train_label.npy', train_label)
    """



def random_shift_flip(image):
    # Random shift values for x and y
    dx, dy = randint(-20, 20), randint(-20, 20)

    # Translation matrix
    M = np.float32([[1, 0, dx], [0, 1, dy]])

    # Image dimensions
    rows, cols = image.shape[:2]

    # Apply the translation
    shifted_image = cv2.warpAffine(image, M, (cols, rows),borderValue=(255, 255, 255))

    # Randomly decide whether to flip the image horizontally
    if choice([True, False]):
        shifted_image = cv2.flip(shifted_image, 1)  # 1 means flipping around y-axis

    return shifted_image



if __name__ == '__main__':
    main()