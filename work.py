#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
import pickle


file_list = []
class_list = []

DATADIR = "../resistor/train"

# All the categories you want your neural network to detect
CATEGORIES = ["100ohm", "200ohm", "300ohm", "1Kohm", "10Kohm","1Mohm"]


# The size of the images that your neural network will use
IMG_SIZE = 40

# Checking or all images in the data folder
for category in CATEGORIES :
	path = os.path.join(DATADIR, category)
	for img in os.listdir(path):
		img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
		new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
		new_array = np.vstack((new_array[:,:,0],new_array[:,:,1],new_array[:,:,2]))

training_data = []

def create_training_data():
	for category in CATEGORIES :
		path = os.path.join(DATADIR, category)
		class_num = CATEGORIES.index(category)
		for img in os.listdir(path):
			try :
				img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
				new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
				new_array = np.vstack((new_array[:,:,0],new_array[:,:,1],new_array[:,:,2]))
				training_data.append([new_array, class_num])
			except Exception as e:
				pass

create_training_data()

random.shuffle(training_data)

X = [] #features
Y = [] #labels

for features, label in training_data:
	X.append(features)
	Y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE*3, IMG_SIZE, 1)
X = X/255.0
Y = np.array(Y)

# Creating the files containing all the information about your model
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()


# In[53]:


X = pickle.load(open("X.pickle", "rb"))
X.shape[1:]


# In[54]:


import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
import pickle


file_list = []
class_list = []

DATADIR = "validation1"

# All the categories you want your neural network to detect
CATEGORIES = ["100ohm", "200ohm", "300ohm", "1Kohm", "10Kohm","1Mohm"]


# The size of the images that your neural network will use
IMG_SIZE = 40

# Checking or all images in the data folder
for category in CATEGORIES :
	path = os.path.join(DATADIR, category)
	for img in os.listdir(path):
		img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
		new_array = np.vstack((img_array[:,:,0],img_array[:,:,1],img_array[:,:,2]))

training_data = []

def create_training_data():
	for category in CATEGORIES :
		path = os.path.join(DATADIR, category)
		class_num = CATEGORIES.index(category)
		for img in os.listdir(path):
			try :
				img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
				new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
				new_array = np.vstack((new_array[:,:,0],new_array[:,:,1],new_array[:,:,2]))
				training_data.append([new_array, class_num])
			except Exception as e:
				pass

create_training_data()

random.shuffle(training_data)
X1 = [] #features
Y1 = [] #labels

for features, label in training_data:
	X1.append(features)
	Y1.append(label)

X1 = np.array(X1).reshape(-1, IMG_SIZE*3, IMG_SIZE, 1)
Y1 = np.array(Y1)

# Creating the files containing all the information about your model
pickle_out = open("X1.pickle", "wb")
pickle.dump(X1, pickle_out)
pickle_out.close()

pickle_out = open("y1.pickle", "wb")
pickle.dump(y1, pickle_out)
pickle_out.close()

#pickle_in = open("X.pickle", "rb")
#X1 = pickle.load(pickle_in)


# In[55]:


import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from keras.models import model_from_json
from keras.models import load_model
import matplotlib.pyplot as plt
from keras import backend as K 


# Opening the files about data
#X = pickle.load(open("X.pickle", "rb"))
#Y = pickle.load(open("y.pickle", "rb"))
#X1 = pickle.load(open("X1.pickle", "rb"))
#Y1 = pickle.load(open("y1.pickle", "rb"))


# normalizing data (a pixel goes from 0 to 255)


# Building the model




model = Sequential()
# 3 convolutional layers
model.add(Conv2D(32, (3, 3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# 2 hidden layers
model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(128))
model.add(Activation("relu"))

# The output layer with 13 neurons, for 13 classes
model.add(Dense(6))
model.add(Activation("softmax"))

# Compiling the model using some basic parameters
model.compile(loss="sparse_categorical_crossentropy",
				optimizer="RMSprop",
				metrics=["accuracy"])

# Training the model, with 40 iterations
# validation_split corresponds to the percentage of images used for the validation phase compared to all the images
#history = model.fit(X, y, batch_size=32, epochs=120, validation_split=0.1)

history = model.fit(X,Y,batch_size=32, epochs=100, verbose=1, validation_data=(X1, Y1))
# Saving the model
model_json = model.to_json()
with open("model1.json", "w") as json_file :
	json_file.write(model_json)

model.save_weights("model1.h5")
print("Saved model to disk")

model.save('CNN1.model')



# Printing a graph showing the accuracy changes during the training phase
print(history.history.keys())
plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')


# In[1]:


def cifar_grid(X,Y,inds,n_col, predictions=None):
    import matplotlib.pyplot as plt
    if predictions is not None:
    if Y.shape != predictions.shape:
        print("Predictions must equal Y in length!\n")
    return(None)
    N = len(inds)
    n_row = int(ceil(1.0*N/n_col))
    fig, axes = plt.subplots(n_row,n_col,figsize=(10,10))
  
    clabels = labels["label_names"]
    for j in range(n_row):
        for k in range(n_col):
            i_inds = j*n_col+k
            i_data = inds[i_inds]
            axes[j][k].set_axis_off()
            if i_inds < N:
                axes[j][k].imshow(X[i_data,...], interpolation="nearest")
                label = clabels[np.argmax(Y[i_data,...])]
                axes[j][k].set_title(label)
                if predictions is not None:
                    pred = clabels[np.argmax(predictions[i_data,...])]
                if label != pred:
                    label += " n"
                    axes[j][k].set_title(pred, color="red")
  
    fig.set_tight_layout(True)
    return fig


# In[3]:


import cv2
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

CATEGORIES = ["100ohm", "200ohm", "300ohm", "1kohm", "10kohm","1mohm"]
CATEGORIES_KEYS = {category_name: category_index for category_index, category_name in enumerate(CATEGORIES)}
COLUMNS_NUM = 6

def prepare(file):
    IMG_SIZE = 40
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
model = tf.keras.models.load_model("CNN.model")

count_val_photos = 0
for _, _, files in os.walk(r'C:\Users\ali88\Downloads\resistor\validation1'):
    count_val_photos += len(files)
# print(count_val_photos)
ROWS_NUM = int(np.ceil(1.0*count_val_photos/COLUMNS_NUM))
fig, axes = plt.subplots(ROWS_NUM, COLUMNS_NUM, figsize=(10,10))

photo_count = 0
for current_dir, _, files in os.walk(r'C:\Users\ali88\Downloads\resistor\validation1'):
    if len(files) > 0:
        current_category = current_dir.split('\\')[-1]
        current_label = CATEGORIES_KEYS[current_category]
    for file in files:
        image = os.path.join(current_dir, file)
        predicted_label = np.argmax(model.predict([prepare(image)])[0])
        is_prediction_correct = current_label == predicted_label
#         print(f'Is prediction correct {is_prediction_correct}')
#         print('------------')
        row = photo_count // COLUMNS_NUM
        column = photo_count % COLUMNS_NUM
#         print((row, column))
        axes[row][column].set_axis_off()
        axes[row][column].imshow(cv2.imread(image, cv2.IMREAD_COLOR), cmap='gray')
        axes[row][column].set_title(current_category)
        
        if not is_prediction_correct:
            axes[row][column].set_title(current_category, color='red')
        photo_count += 1

# prediction = model.predict([prepare(image)])
# prediction = list(prediction[0])
# print(CATEGORIES[prediction.index(max(prediction))])


# In[3]:


import os
import cv2
cv2.IMREAD_GRAYSCALEcv.IMREAD_COLORcv.IMREAD_COLOR
for current_dir, dirs, files in os.walk(r'C:\Users\ali88\Downloads\resistor\validation1'):
    print(tmp)

