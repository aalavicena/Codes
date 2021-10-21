#Load necessary inputs
from re import I
from librosa.feature.spectral import chroma_stft, spectral_centroid
import pandas as pd
import os 
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pickle
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
import tensorflow as tf

#Import dataframe
os.chdir('G:/Datasets/Bird Presence/warblrb10k_public_wav')
df = pd.read_csv('warblrb10k_public_metadata_2018.csv')
df.head()

#Load the audio file into visual cues using Librosa
sample_num=3 #pick a file to display
#get the audio file
filename=df.itemid[sample_num]+str('.wav')
#do I need to define the time signature?
#load the file
y,sr=librosa.load('wav/'+str(filename))
librosa.display.waveplot(y,sr=sr, x_axis='time', color='cyan')

#Make the audio features looked like an image
def padding (array, xx, yy):
    """
    :param array:numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """
    h=array.shape[0]
    w=array.shape[1]

    a=max((xx-h)//2,0)
    aa=max(0,xx-a-h)
    b=max(0,(yy-w)//2)
    bb=max(yy-b-w,0)
    
    return np.pad(array, pad_width=((a,aa),(b,bb)), mode ='constant')

#Extracting features: MFCC, spectral bandwidth, spectral centroid, chromagram, short-time Fourier transform
def generate_features(y_cut):
    max_size=1000
    stft=padding(np.abs(librosa.stft(y_cut,n_fft=255,hop_length=512)),128,max_size)
    MFCCs=padding(librosa.feature.mfcc(y_cut,n_fft=255,hop_length=512,n_mfcc=128),128,max_size)
    spectral_centroid=librosa.feature.spectral_centroid(y=y_cut,sr=sr)
    chroma_stft=librosa.feature.chroma_stft(y=y_cut,sr=sr)
    spec_bw=librosa.feature.spectral_bandwidth(y=y_cut,sr=sr)

    #pad into shape
    image=np.array([padding(normalize(spec_bw),1,max_size)]).reshape(1,max_size)
    image=np.append(image,padding(normalize(spectral_centroid),1,max_size),axis=0)

    #repeat the padded spec_bw,spec_centroid and chroma stft until they are stft and MFCC-sized
    for i in range (0,9):
        image=np.append(image,padding(normalize(spec_bw),1,max_size),axis=0)
        image=np.append(image,padding(normalize(spectral_centroid),1,max_size),axis=0)
        image=np.append(image,padding(normalize(chroma_stft),12,max_size),axis=0)
        image=np.dstack((image,np.abs(stft)))
        image=np.dstack((image,MFCCs))
        return image
X=df.drop('hasbird',axis=1)
y=df.hasbird 

#extract training, test, and validation sets
#split once to get test and training set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=123,stratify=y)
print(X_train.shape,X_test.shape)

#split twice to get validation set
X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.25,random_state=123)
print(X_train.shape,X_test.shape,X_val.shape,len(y_train),len(y_test),len(y_val))

#calculate these features for each audio file
def get_features(df_in):
    features=[]
    labels=[] #empty array to store labels
    #for each species, determine how many augmentations are needed
    df_in=df_in.reset_index()
    for i in df_in.hasbird.unique():
        print('hasbird',I)
        #all the file indices with the same hasbird
        filelist=df_in.loc[df_in.hasbird == I].index
    for j in range(0,len(filelist)):
        filenamme=df_in.iloc[filelist[j]].itemid
        +str('.wav') #get the filename
        itemid=df_in.iloc[filelist[j]].itemid
        hasbird=I
        #load the file
        y,sr=librosa.load(filename,sr=28000)
        #cut the file to signal start and end (?)
        y_cut=y[int(round(*sr)):int(round(*sr))]
        #generate features and output numpy array
        data=generate_features(y_cut)
        features.append(data[np.newaxis,...])
        labels.append(hasbird)
    output=np.concatenate(features,axis=0)
    return(np.array(output,labels))

#use get_features to calculate and store features
test_features,test_labels=get_features(pd.concat([X_test,y_test],axis=1))
train_features,train_labels=get_features(pd.concat([X_train,y_train],axis=1))
#normalize the data and cast into numpy array
X_train=np.array((X_train-np.min(X_train))/(np.max(X_train)-np.min(X_train)))
X_test=np.array((X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test)))
X_train=X_train/np.std(X_train)
X_test=X_test/np.std(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)

#build a CNN
input_shape=(128,1000,3)
CNNmodel=models.Sequential()
CNNmodel.add(layers.Conv2D(32,(3.3),activation='relu',input_shape=input_shape))
CNNmodel.add(layers.MaxPooling2D((2,2)))
CNNmodel.add(layers.Dropout(0.2))
CNNmodel.add(layers.Conv2D(64,(3,3),activation='relu'))
CNNmodel.add(layers.MaxPooling2D((2,2)))
CNNmodel.add(layers.Dropout(0.2))
CNNmodel.add(layers.Conv2D(64,(3,3),activation='relu'))
CNNmodel.add(layers.Flatten())
CNNmodel.add(layers.Dense(64,activation='relu'))
CNNmodel.add(layers.Dropout(0.2))
CNNmodel.add(layers.Dense(32,activation='relu'))
CNNmodel.add(layers.Dense(24,activation='softmax'))

#compile the model
CNNmodel.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),metrics=['accuracy'])
#fit the model
history=CNNmodel.fit(X_train,y_train,epochs=20,validation_data=(X_val,y_val))
#evaluate the model
history_dict=history.history
loss_values=history_dict['loss']
acc_values=history_dict['accuracy']
val_loss_values=history_dict['val_loss']
val_acc_values=history_dict['val_accuracy']
epochs=range(1,21)
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))
ax1.plot(epochs,loss_values,'bo',label='Training Loss')
ax1.plot(epochs,val_loss_values,'orange',label='Validation Loss')
ax1.set_title('Training and Validation Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()
ax2.plot(epochs,acc_values,'bo',label='Training Accuracy')
ax2.plot(epochs,val_acc_values,'orange',label='Validation Accuracy')
ax2.set_title('Training and Validation Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()
plt.show()

#end