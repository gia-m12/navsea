#from navsea.corrosion.classify import CorrosionClassfier
from navsea.corrosion import CorrosionClassifier
from cv2 import imshow
import matplotlib.pyplot as plt
if __name__ == "__main__":
    testobj = CorrosionClassifier()
    testobj.load_weightfiles("/home/jetbot/pythonAWS/navsea/model/best_UNet_model.h5")
    output=testobj.predict("image_25.png")
    plt.imsave('test1.png', output, cmap="Blues")
    #imshow(output)