import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

class DenoisenetModel:
    def __init__(self):
        self.sess = tf.compat.v1.Session()

    def predict(self, mdl_dir, output_dir, input_img_str):
        input_img = cv2.cvtColor(cv2.imread(input_img_str), cv2.COLOR_BGR2RGB)
        assert input_img.shape == (400, 400,3)
        _ = tf.compat.v1.saved_model.load(self.sess, ["serve"], mdl_dir)
        g = tf.compat.v1.get_default_graph()
        x = g.get_tensor_by_name('input/x:0')
        cnn_out = g.get_tensor_by_name('out/dn_img:0')
        input_img_padded = np.zeros((1, 450, 450, 3))
        input_img_padded[0, 25:425, 25:425, :] = input_img
        output_img = self.sess.run(cnn_out, feed_dict={x: input_img_padded})
        output_img = output_img[0, 25:425, 25:425, :]
        output_img /= np.max(output_img)
        output_img = (output_img*255.0).astype(np.uint8)
        plt.imsave(output_dir + '/denoised_img.png', output_img)

    def train(self, noisy_data, ref_data):
        _ = tf.compat.v1.saved_model.load(self.sess, ["serve"], mdl_dir)
        g = tf.compat.v1.get_default_graph()
        train_step = g.get_operation_by_name("out/train")
        x = g.get_tensor_by_name("input/x:0")
        x_ref = g.get_tensor_by_name("input/x_ref:0")
        sess.run(train_step, feed_dict={x: noisy_data, x_ref: ref_data})
