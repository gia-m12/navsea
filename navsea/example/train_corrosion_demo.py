from navsea.corrosion import CorrosionClassifier


if __name__ == "__main__":
    trainobj = CorrosionClassifier()
    trainobj.train(imagefolder='im', epochs=50)
    trainobj.savewgtfile(wgtfileloc='../model')
   # output=trainobj.train()

#images are located in the examples file folder



