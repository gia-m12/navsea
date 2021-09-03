from navsea.biofouling import BiofoulingClassfier


if __name__ == "__main__":
    trainobj = BiofoulingClassfier()
    output=trainobj.train()

#images are located in the examples file folder



