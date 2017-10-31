
import numpy as np 
import pandas as pd 
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py

import keras
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization,Flatten
from keras.utils import to_categorical

# Any results you write to the current directory are saved as output.


files = glob('/home/miso/Downloads/train/*.jpg')
labels = pd.read_csv('labels.csv')
labels.head()

Image.open(files[0])

labels.describe()

df = labels.breed.value_counts().reset_index()
df[-10:]

df.describe()

imgs = np.array([np.array(Image.open(file).resize((90,90))) for file in files])
ids = [file.split('/')[-1].split('.')[0] for file in files]
id_val = dict(zip(labels.id,labels.breed))
label_ids = [id_val[id] for id in ids]
label_index = {label:i for i,label in enumerate(np.unique(label_ids))}
labels_id = [label_index[label] for label in label_ids]

labels_onehot_encoded = to_categorical(labels_id,num_classes=120)

base_model = VGG16(include_top=False,weights='imagenet',input_shape=(90,90,3))

model = Sequential(base_model.layers[:7])
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(120,activation='softmax'))
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(imgs,labels_onehot_encoded,batch_size=8,validation_split=0.2,)

np.array(imgs).shape
