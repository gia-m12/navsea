from navsea.biofouling import BiofoulingClassfier
from cv2 import imshow
import matplotlib.pyplot as plt
if __name__ == "__main__":
    testobj = BiofoulingClassfier()
    testobj.load_weightfiles("../model/best_Unet_biofouling_model.h5")
    output=testobj.predict("image_26.png")
    plt.imsave('test1_b.png', output, cmap="Blues")
    #imshow(output)