import numpy as np
import scipy.io as sio
import math
import time
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy
from matplotlib import cm


dataset=[]
k=4
def loadDatadet(infile,k):
    f=open(infile,'r')
    sourceInLine=f.readlines()
    dataset=[]
    for line in sourceInLine:
        temp1=line.strip('\n')
        temp2=temp1.split('\t')
        dataset.append(temp2)
    return dataset
def normalization(x):
    y=np.array(x)
    Max=max(y)
    Min=min(y)
    for i in range(len(x)):
        y[i]=(y[i]-Min)/(Max-Min)
    return y.tolist()
    # return x

#
p=400
p1=p*10
#----------------------crack 0-------------------
mj=0
temp00=[]
infile=r'Gearbox\Dataset_1\LW-00\1500-0.txt'
temp1=loadDatadet(infile,k)
temp1=temp1[1:p1+1]
for i in range(0,len(temp1)):
    temp00.append(float(temp1[i][k]))
temp00=np.array(temp00)
temp00=normalization(temp00)
for j in range(0,p1,p):
    dataset.append(temp00[0+j:p+j])


temp01=[] 
infile=r'Gearbox\Dataset_1\LW-00\1500-2.txt'
temp1=loadDatadet(infile,k)
temp1=temp1[1:p1+1]
for i in range(0,len(temp1)):
    temp01.append(float(temp1[i][k]))
temp01=np.array(temp01)
temp01=normalization(temp01)
for j in range(0,p1,p):
    dataset.append(temp01[0+j:p+j])


infile=r'Gearbox\Dataset_1\LW-00\1500-4.txt'
temp02=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:p1+1]
for i in range(0,len(temp1)):
    temp02.append(float(temp1[i][k]))
temp02=np.array(temp02)
temp02=normalization(temp02)
for j in range(0,p1,p):
    dataset.append(temp02[0+j:p+j])

    
infile=r'Gearbox\Dataset_1\LW-00\1500-6.txt'
temp03=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:p1+1]
for i in range(0,len(temp1)):
    temp03.append(float(temp1[i][k]))
temp03=np.array(temp03)
temp03=normalization(temp03)
for j in range(0,p1,p):
    dataset.append(temp03[0+j:p+j])


infile=r'Gearbox\Dataset_1\LW-00\1500-8.txt'
temp04=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:p1+1]
for i in range(0,len(temp1)):
    temp04.append(float(temp1[i][k]))
temp04=np.array(temp04)
temp04=normalization(temp04)
for j in range(0,p1,p):
    dataset.append(temp04[0+j:p+j])

    
infile=r'Gearbox\Dataset_1\LW-00\1500-10.txt'
temp05=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:p1+1]
for i in range(0,len(temp1)):
    temp05.append(float(temp1[i][k]))
temp05=np.array(temp05)
temp05=normalization(temp05)
for j in range(0,p1,p):
    dataset.append(temp05[0+j:p+j])

     
# #----------------------crack 5-------------------
mj=1
temp10=[]
infile=r'Gearbox\Dataset_1\LW-01\1500-0.txt'
temp1=loadDatadet(infile,k)
temp1=temp1[1:p1+1]
for i in range(0,len(temp1)):
    temp10.append(float(temp1[i][k]))
temp10=np.array(temp10)
temp10=normalization(temp10)
for j in range(0,p1,p):
    dataset.append(temp10[0+j:p+j])


temp11=[] 
infile=r'Gearbox\Dataset_1\LW-01\1500-2.txt'
temp1=loadDatadet(infile,k)
temp1=temp1[1:p1+1]
for i in range(0,len(temp1)):
    temp11.append(float(temp1[i][k]))
temp11=np.array(temp11)
temp11=normalization(temp11)
for j in range(0,p1,p):
    dataset.append(temp11[0+j:p+j])


infile=r'Gearbox\Dataset_1\LW-01\1500-4.txt'
temp12=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:p1+1]
for i in range(0,len(temp1)):
    temp12.append(float(temp1[i][k]))
temp12=np.array(temp12)
temp12=normalization(temp12)
for j in range(0,p1,p):
    dataset.append(temp12[0+j:p+j])

    
infile=r'Gearbox\Dataset_1\LW-01\1500-6.txt'
temp13=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:p1+1]
for i in range(0,len(temp1)):
    temp13.append(float(temp1[i][k]))
temp13=np.array(temp13)
temp13=normalization(temp13)
for j in range(0,p1,p):
    dataset.append(temp13[0+j:p+j])


infile=r'Gearbox\Dataset_1\LW-01\1500-8.txt'
temp14=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:p1+1]
for i in range(0,len(temp1)):
    temp14.append(float(temp1[i][k]))
temp14=np.array(temp14)
temp14=normalization(temp14)
for j in range(0,p1,p):
    dataset.append(temp14[0+j:p+j])

    
infile=r'Gearbox\Dataset_1\LW-01\1500-10.txt'
temp15=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:p1+1]
for i in range(0,len(temp1)):
    temp15.append(float(temp1[i][k]))
temp15=np.array(temp15)
temp15=normalization(temp15)
for j in range(0,p1,p):
    dataset.append(temp15[0+j:p+j])

    
# #----------------------crack 10-------------------
mj=2
temp20=[]
infile=r'Gearbox\Dataset_1\LW-02\1500-0.txt'
temp1=loadDatadet(infile,k)
temp1=temp1[1:p1+1]
for i in range(0,len(temp1)):
    temp20.append(float(temp1[i][k]))
temp20=np.array(temp20)
temp20=normalization(temp20)
for j in range(0,p1,p):
    dataset.append(temp20[0+j:p+j])
 

temp21=[] 
infile=r'Gearbox\Dataset_1\LW-02\1500-2.txt'
temp1=loadDatadet(infile,k)
temp1=temp1[1:p1+1]
for i in range(0,len(temp1)):
    temp21.append(float(temp1[i][k]))
temp21=np.array(temp21)
temp21=normalization(temp21)
for j in range(0,p1,p):
    dataset.append(temp21[0+j:p+j])


infile=r'Gearbox\Dataset_1\LW-02\1500-4.txt'
temp22=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:p1+1]
for i in range(0,len(temp1)):
    temp22.append(float(temp1[i][k]))
temp22=np.array(temp22)
temp22=normalization(temp22)
for j in range(0,p1,p):
    dataset.append(temp22[0+j:p+j])

    
infile=r'Gearbox\Dataset_1\LW-02\1500-6.txt'
temp23=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:p1+1]
for i in range(0,len(temp1)):
    temp23.append(float(temp1[i][k]))
temp23=np.array(temp23)
temp23=normalization(temp23)
for j in range(0,p1,p):
    dataset.append(temp23[0+j:p+j])


infile=r'Gearbox\Dataset_1\LW-02\1500-8.txt'
temp24=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:p1+1]
for i in range(0,len(temp1)):
    temp24.append(float(temp1[i][k]))
temp24=np.array(temp24)
temp24=normalization(temp24)
for j in range(0,p1,p):
    dataset.append(temp24[0+j:p+j])

    
infile=r'Gearbox\Dataset_1\LW-02\1500-10.txt'
temp25=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:p1+1]
for i in range(0,len(temp1)):
    temp25.append(float(temp1[i][k]))
temp25=np.array(temp25)
temp25=normalization(temp25)
for j in range(0,p1,p):
    dataset.append(temp25[0+j:p+j])

    
# #----------------------crack 15-------------------
mj=3
temp30=[]
infile=r'Gearbox\Dataset_1\LW-03\1500-0.txt'
temp1=loadDatadet(infile,k)
temp1=temp1[1:p1+1]
for i in range(0,len(temp1)):
    temp30.append(float(temp1[i][k]))
temp30=np.array(temp30)
temp30=normalization(temp30)
for j in range(0,p1,p):
    dataset.append(temp30[0+j:p+j])


temp31=[] 
infile=r'Gearbox\Dataset_1\LW-03\1500-2.txt'
temp1=loadDatadet(infile,k)
temp1=temp1[1:p1+1]
for i in range(0,len(temp1)):
    temp31.append(float(temp1[i][k]))
temp31=np.array(temp31)
temp31=normalization(temp31)
for j in range(0,p1,p):
    dataset.append(temp31[0+j:p+j])


infile=r'Gearbox\Dataset_1\LW-03\1500-4.txt'
temp32=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:p1+1]
for i in range(0,len(temp1)):
    temp32.append(float(temp1[i][k]))
temp32=np.array(temp32)
temp32=normalization(temp32)
for j in range(0,p1,p):
    dataset.append(temp32[0+j:p+j])

    
infile=r'Gearbox\Dataset_1\LW-03\1500-6.txt'
temp33=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:p1+1]
for i in range(0,len(temp1)):
    temp33.append(float(temp1[i][k]))
temp33=np.array(temp33)
temp33=normalization(temp33)
for j in range(0,p1,p):
    dataset.append(temp33[0+j:p+j])


infile=r'Gearbox\Dataset_1\LW-03\1500-8.txt'
temp34=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:p1+1]
for i in range(0,len(temp1)):
    temp34.append(float(temp1[i][k]))
temp34=np.array(temp34)
temp34=normalization(temp34)
for j in range(0,p1,p):
    dataset.append(temp34[0+j:p+j])

    
infile=r'Gearbox\Dataset_1\LW-03\1500-10.txt'
temp35=[]
temp1=loadDatadet(infile,k)
temp1=temp1[1:p1+1]
for i in range(0,len(temp1)):
    temp35.append(float(temp1[i][k]))
temp35=np.array(temp35)
temp35=normalization(temp35)
for j in range(0,p1,p):
    dataset.append(temp35[0+j:p+j])


s=[]
freqs=[]
for i in range(len(dataset)):
    y2=dataset[i]
    s.append(y2)

y=[] #label

for i in range(60):
    y.append(0)

for i in range(60):
    y.append(1)
    
for i in range(60):
    y.append(2)
    
for i in range(60):
    y.append(3)

image=s
im=np.array(image)
im.shape=240,20,20,1  
image=im

label=y

import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten,LeakyReLU,Dropout,GlobalMaxPooling2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from skimage import util
data=image
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( data, label, test_size=0.4, random_state=42,stratify = label) #divide train,test,validation



def build_CNN(): #CNN model
    model=Sequential()

    model.add(Convolution2D(
        batch_input_shape=(None,20,20,1),     
        filters=16,                          
        kernel_size=5,                       
        strides=1,                       
        padding='same',                       
        ))

    model.add(LeakyReLU(alpha=0.3))          
    model.add(MaxPooling2D(
    pool_size=2,         
    strides=1,            
    padding='same',  
    )) 
    model.add(Dropout(0.4))

    model.add(Convolution2D(8,8,strides=2,padding='same'))
    model.add(LeakyReLU(alpha=0.3))  
    model.add(MaxPooling2D(2,2,'same'))
    model.add(Dropout(0.4))

    model.add(Convolution2D(8,8,strides=2,padding='same'))
    model.add(LeakyReLU(alpha=0.3))  
    model.add(MaxPooling2D(2,2,'same'))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(200))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(4))        
    model.add(Activation('softmax'))  
    adam=Adam(lr=1e-3)      
    model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])                
    return model


epochs=100
batch_size=16
test_acc=[]
x_train = np.array(X_train).reshape(-1, 20,20,1).astype('float32')
x_test = np.array(X_test).reshape(-1, 20,20,1).astype('float32')
y_train=np_utils.to_categorical(Y_train,num_classes=4)
y_test=np_utils.to_categorical(Y_test,num_classes=4)
x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean
model=build_CNN()
model.summary()
acc=[]

for i in range(10):  #diagnosis
    history = model.fit(x_train, y_train, validation_split=0.5, epochs=epochs, batch_size=batch_size, verbose=1)
    loss,accuracy=model.evaluate(x_test,y_test) 
    acc.append([accuracy])
    print(accuracy*100)





