# importing required libraries those are necessary for our model

#preprocessing purpose
import cv2
import numpy as np
import os
#----------

import tensorflow as tf
from tensorflow.keras.models import Sequential            # it sets the all layers in a Sequence (layer by layer model)
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout   #different types of layers we need in our model
from tensorflow.keras.preprocessing.image import ImageDataGenerator # it generates different images by rotating, normalizing etc an image for variation it helps in better training of our model
from sklearn.model_selection import train_test_split #splits our dataset in training and testing datasets
from sklearn.preprocessing import LabelBinarizer #converts class labels into binary format (one hot encoding)
#one hot encoding means the image belongs to a class is 1 and other classes than that class are zero(0,1,0,0,0) for 2nd class
import matplotlib.pyplot as plt # plotting the performance of the model to analyze model performance by visualizing it

# preprocessing the data

#array which stores the path of the images
images_path = []
# loading dataset
dataset_path="D:\HandGestureRecognition\leapGestRecog"
# class names
classes = ["01_palm","02_I","03_Fist","04_Fist_Moved","05_Thumb","06_Index","07_Ok","08_Palm_moved","09_C","10_Down"]

for root,dirs,files in os.walk(dataset_path,topdown=False):
    for name in files:
        path=os.path.join(root,name)
        if path.endswith("png"):
            images_path.append(path)

# images and labels dataset
images = []
labels = []

for image in images_path:
    # reading every image one by one and converting it into grayscale image
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    # resizing it as the required input shape for our CNN  model
    resized_img = cv2.resize(img, (64, 64))

    images.append(resized_img)

    # processing labels for images
    temp = image.split("\\")[4]
    temp = temp[3:]
    # adding image label to the label list
    labels.append(temp)

# creating numpy arrays of images and labels lists
X = np.array(images)
y = np.array(labels)

# adding channel to images (1 for gray scale and 3 for RGB images)
# in our case it is grayscale
X = X.reshape(X.shape[0],64,64,1)

# normalizing pixel values
X = X/255.0
print(f"Dataset loaded with {X.shape[0]} images.")

# train test split
# test_size=0.2 means 20% test and 80% training data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)


# convert label to one hot encoding
lb = LabelBinarizer() # creating instance of LabelBinarizer class
y_train = lb.fit_transform(y_train)  # firstly we use fit_transform so that binarizer can learn pattern from the dataset
y_test = lb.transform(y_test)   # after we can use only transform (by applying the learnt patterns from before fit_transform methood)

# applying data augmentation

# initializing ImageDataGenerator with some settings
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=10,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

# applying  augmentation in batches
augmented_data = datagen.flow(X_train,y_train,batch_size=32, subset = "training")
validation_dataa = datagen.flow(X_test,y_test,batch_size=32, subset = "validation")

x_batch,y_batch = next(augmented_data)
print(x_batch.shape,y_batch.shape)

# building the CNN model
model = Sequential()

model.add(Conv2D(32,(3,3),input_shape = (64,64,1),activation="relu"))   # 32 filters of 3x3 size
model.add(MaxPooling2D(pool_size=(2,2)))  # max pooling layer of 2x2 size

model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(128,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

#flatten layer
model.add(Flatten())

#fully connected layer
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))  # to prevent ovevrfitting

#output layer
model.add(Dense(10, activation = "softmax"))


model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics=['accuracy'])

my_model = model.fit(augmented_data, validation_data = validation_dataa, epochs=20)

# checking the accuracy of the model

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# checking the accuracy and loss by the graph of the model

history = my_model
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()