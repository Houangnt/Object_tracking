import os
os.getcwd()
collection = "C:/Users/GHTK/Downloads/5/labels"
for i, filename in enumerate(os.listdir(collection)):
    os.rename("C:/Users/GHTK/Downloads/5/labels/" + filename, "C:/Users/GHTK/Downloads/5/labels/00" + str(1000+i) + ".jpg")