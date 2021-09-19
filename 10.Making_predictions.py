import keras
from keras.datasets import cifar10
from keras.models import Sequential, model_from_json
from keras.preprocessing import image
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

class_labels = {
    0: "Plane",
    1: "car",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Boat",
    9: "Truck"
}

f = Path("model_structure.json")
model_structure = f.read_text()
model = model_from_json(model_structure)
model.load_weights("model_weights.h5")
img = image.load_img("cat.png", target_size=(32,32))
image_to_test = image.img_to_array(img) / 255
list_of_images = np.expand_dims(image_to_test, axis=0)

results = model.predict(list_of_images)
single_result = results[0]
most_likely_class_index = int(np.argmax(single_result))
class_likelihood  = single_result[most_likely_class_index]
class_label = class_labels[most_likely_class_index]

print("This image is a {} - Likelihood: {:2f}".format(class_label, class_likelihood) )
