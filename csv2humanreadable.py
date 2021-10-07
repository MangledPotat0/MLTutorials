import csv
import os
import numpy as np
from  PIL import Image
import json
import pathlib
basedir  = "C:\\Users\\user\\Desktop\\Coding\\pythons\\MLtutorials\\"

for tagname in ["test","train"]:
    testdir  = os.path.join(basedir,tagname)
    pathlib.Path(testdir).mkdir(parents=True, exist_ok=True)
    with open("mnist_{}.csv".format(tagname)) as fh:
        csvfh = csv.reader(fh)
        idx = 0
        for line in csvfh:
            label = line[0]
            dirtoplace = os.path.join(testdir,label)
            pathlib.Path(dirtoplace).mkdir(parents=True, exist_ok=True)
            imagearr = np.array([int(p) for p in line[1:]],dtype = np.uint8)
            imagearr = np.reshape(imagearr,(28,28))

            im = Image.fromarray(imagearr)
            im.save(os.path.join(dirtoplace,"{}.jpeg".format(idx)))
            idx += 1
    
