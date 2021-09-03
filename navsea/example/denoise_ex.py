#!/usr/bin/env python

from navsea.denoisenet import DenoisenetModel

if __name__ == "__main__":
    cnn_mdl = DenoisenetModel()
    # output image will be written to out/denoised_img.png
    cnn_mdl.predict(mdl_dir='mdl', output_dir='out', input_img_str='test_img_noisy.png')
