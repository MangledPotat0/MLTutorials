import pickle
import keras
import numpy as np
from PIL import Image

with open("history.pkl","rb") as fh:
    history = pickle.load(fh)
    print(history)

def runontestimage(testimage):
    dd = keras.models.load_model("pretraineddiscriminator.mdl")
    data = Image.open(testimage)
    data = np.reshape(data,(1,28,28,1))
    output = dd.predict(data)
    print(np.argmax(output))

runontestimage('yourface1000.png')

#EOF
