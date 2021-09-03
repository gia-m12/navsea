import random

import cv2
import numpy as np


def load_one_image(image_path, image_size=128):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))  # resizing before inserting to the network



    #normalize image
    #image = image / 255.0

    return image


def generate_underwater_img(img, distance, medias):
    img = img
    depth_array = np.ones((128, 128)) * distance
    underwater_imgs = []
    background = random.uniform(0.8, 1)

    for media in medias:
        energy_ratio = []
        for beta in media:
            ratio = np.power(beta, depth_array)
            energy_ratio.append(ratio)
        energy_ratio = np.transpose(np.array(energy_ratio), (1, 2, 0))
        underwater = np.multiply(img, energy_ratio) + background * (1 - energy_ratio)
        underwater = underwater / 255
        underwater_imgs.append(underwater)

    return underwater_imgs


def data_augmentation(orig_img, label, medias):
    all_img = [orig_img/255]
    flip_1 = cv2.flip(orig_img, 0)/255
    flip_2 = cv2.flip(orig_img, 1)/255
    flip_3 = cv2.flip(orig_img, -1)/255
    rotate_1 = cv2.rotate(orig_img, cv2.ROTATE_90_CLOCKWISE)/255
    rotate_2 = cv2.rotate(orig_img, cv2.ROTATE_90_COUNTERCLOCKWISE)/255

    all_img.append(flip_1)
    all_img.append(flip_2)
    all_img.append(flip_3)
    all_img.append(rotate_1)
    all_img.append(rotate_2)

    if label is False:
        blur_img = cv2.blur(orig_img,(10, 10)) / 255
        gaussian = np.random.normal(0, 0.4, (orig_img.shape[0], orig_img.shape[1])).reshape(128, 128, 1)
        noisy_img = orig_img / 255 + gaussian
        noisy_img = np.clip(noisy_img, 0, 1)
        all_img.append(blur_img.reshape(128, 128, orig_img.shape[2]))
        all_img.append(noisy_img.reshape(128, 128, orig_img.shape[2]))
        underwater_imgs_1 = generate_underwater_img(orig_img, distance=0.5, medias = medias)
        underwater_imgs_2 = generate_underwater_img(orig_img, distance=1.2, medias = medias)
        all_img += underwater_imgs_1
        all_img += underwater_imgs_2
    else:
        all_img.append(orig_img/255)
        all_img.append(orig_img/255)
        all_img += [orig_img/255] * (len(medias) * 2)

    return all_img