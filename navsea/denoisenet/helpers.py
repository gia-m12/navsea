import tensorflow as tf
import numpy as np
from PIL import Image

def _convert_to_grayscale(images):
    # courtesy of https://stackoverflow.com/a/12201744
    return np.dot(images[...,:3], [0.2989, 0.5870, 0.1140])

def _data_from_img_file(fname):
    new_image = np.array(Image.open(fname).resize((400,400)))
    if new_image.shape[-1] == 3:
        new_image = _convert_to_grayscale(new_image).astype(np.uint8)
    else:
        new_image = new_image[:,:,0]
    new_image = np.expand_dims(new_image, 0)
    new_fname = fname.split('/')[-1]
    return (new_image, [new_fname])

def _data_from_directory(dname):
    # using glob for multiple filetypes: https://stackoverflow.com/a/3450732
    image_files = glob.glob(dname + '/*.png') + glob.glob(dname + '/*.jpg') + \
        glob.glob(dname + '/*.jpeg') + glob.glob(dname + '/*.PNG') + \
        glob.glob(dname + '/*.JPG') + glob.glob(dname + '/*.JPEG')
    new_images = np.zeros((1,400,400)).astype(np.uint8)
    for fname in image_files:
        new_image = _data_from_img_file(fname)[0]
        new_images = np.concatenate([new_images, new_image])
    new_images = new_images[1:]
    fnames = [x.split('/')[-1] for x in image_files]
    return (new_images, fnames)

def _process_input(args):
    images = np.zeros((1,400,400)).astype(np.uint8)
    output_filenames = []
    for arg in args:
        # argument is the name of a single image
        if os.path.isfile(arg):
            new_image, new_fname = _data_from_img_file(arg)
            images = np.concatenate([images, new_image])
            output_filenames.extend(new_fname)
        # argument is a directory
        elif os.path.isdir(arg):
            new_images, new_fnames = _data_from_directory(arg)
            images = np.concatenate([images, new_images])
            output_filenames.extend(new_fnames)
    images = images[1:].astype(np.float32) / 255.0
    images_padded = np.zeros((images.shape[0],450,450,1)).astype(np.float32)
    images_padded[:,25:425,25:425,0] = images
    return images_padded, output_filenames

