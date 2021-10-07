from keras.models import Sequential
from keras.layers import Conv2D, Dense,Flatten,MaxPooling2D
from tensorflow.keras.preprocessing import image_dataset_from_directory
import keras
from PIL import Image
import numpy as np
import pickle

def create_digit_detector():

    digitdetector = Sequential(name = 'asdf')
    digitdetector.add(Conv2D(6, (5, 5), activation = "relu",
                      input_shape = (28, 28, 1),name = "5"))

    digitdetector.add(MaxPooling2D(pool_size=(2,2)))
    
    digitdetector.add(Conv2D(16, (5, 5), activation = "relu",
                      name = "6",padding = "same"))
    
    digitdetector.add(MaxPooling2D(pool_size = (2, 2)))

    digitdetector.add(Flatten())
    
    digitdetector.add(Dense(120, activation='tanh', name="8"))
    digitdetector.add(Dense(100, activation='tanh', name="9"))
    digitdetector.add(Dense(10, activation='sigmoid', name="10"))

    digitdetector.compile(loss = 'binary_crossentropy',
                          optimizer = "Adam")

    return(digitdetector)


if __name__ == '__main__':

    dd = create_digit_detector()

    #Telling it what sequence we are using to do training/validation
    trainingset = image_dataset_from_directory(
                        "stuff/train",
                        image_size = (28, 28),
                        batch_size = 16,
                        color_mode = "grayscale",
                        label_mode = "categorical")

    validationset = image_dataset_from_directory(
                        "stuff/test",
                        image_size = (28, 28),
                        batch_size = 16,
                        color_mode = "grayscale",
                        label_mode = "categorical")

    print(dd.summary())

    hist = dd.fit(
            x = trainingset,
            epochs = 1,
            validation_data = validationset
            )

    with open("history.pkl","wb+") as fh:
        pickle.dump(hist.history,fh)
    dd.save("pretraineddiscriminator.mdl")






# EOF
