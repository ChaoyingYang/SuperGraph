import psutil
import os
import numpy as np
import scipy.io as sio
import math
import time
from scipy.fftpack import fft, fftshift, ifft
from scipy.fftpack import fftfreq
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import scipy
from matplotlib import cm




def read_mat(path,key):  
    data= sio.loadmat(path) 
    return(data[key])


def splitlist(list1):   
    alist=[]
    a=0
    for sublist in list1:
        try:
            for i in sublist:
                alist.append(i)
        except TypeError:
            alist.append(sublist)
    for i in alist:
        if type(i)==type([]):
            a=+1
            break
    if  a==1:
        return splitlist(alist)
    if a==0:
        return alist

def arr_size(arr,size):  
    s=[]
    for i in range(0,int(len(arr))+1,size):
        c=arr[i:i+size]
        s.append(c)
    return s

#数据准备
#---------------------------0 load--------------------------------
path='KAT/KATData0.mat'
key='data'
data01=read_mat(path,key)
data01=data01.tolist()
datas01=data01[0:10]+data01[100:110]+data01[200:210]
key='label'
y01=read_mat(path, key)
y01=splitlist(y01)
ys01=y01[0:10]+y01[100:110]+y01[200:210]

#---------------------------1 load--------------------------------
path='KAT/KATData1.mat'
key='data'
data02=read_mat(path,key)
data02=data02.tolist()
datas02=data02[0:10]+data02[100:110]+data02[200:210]
key='label'
y02=read_mat(path, key)
y02=splitlist(y02)
ys02=y02[0:10]+y02[100:110]+y02[200:210]

#---------------------------2 load--------------------------------
path='KAT/KATData2.mat'
key='data'
data03=read_mat(path,key)
data03=data03.tolist()
datas03=data03[0:10]+data03[100:110]+data03[200:210]
key='label'
y03=read_mat(path, key)
y03=splitlist(y03)
ys03=y03[0:10]+y03[100:110]+y03[200:210]

#---------------------------3 load--------------------------------
path='KAT/KATData3.mat'
key='data'
data04=read_mat(path,key)
data04=data04.tolist()
datas04=data04[0:10]+data04[100:110]+data04[200:210]
key='label'
y04=read_mat(path, key)
y04=splitlist(y04)
ys04=y04[0:10]+y04[100:110]+y04[200:210]

y=[]
for x in ys01:
    y.append(x)
for x in ys02:
    y.append(x)
for x in ys03:
    y.append(x)
for x in ys04:
    y.append(x)
y=[x-1 for x in y]  
 
dataa=[]
for i in datas01:
    dataa.append(i)
for i in datas02:
    dataa.append(i)
for i in datas03:
    dataa.append(i)
for i in datas04:
    dataa.append(i)



import scipy.io as sio
import scipy.signal as signal
import operator as opt
def diedai(a):
    Train = []
    Test = []
    Train_label = []
    Test_label = []
    arr = np.arange(len(a))
    np.random.shuffle(arr)
    m = -1  
    for i in arr:
        m+=1
        if m<0.1*len(arr):
            Train.append(a[i])
            Train_label.append(y[i])
        else:
            Test.append(a[i])
            Test_label.append(y[i]) 
    return Train,Train_label,Test,Test_label


def kNN(dataset, labels, testdata, k):
    distSquareMat = []
    for i in range(len(dataset)):
        distSquareMat.append(np.sqrt(sum(np.square(dataset[i] - testdata))))
    distSquareMat = np.array(distSquareMat) 
    sortedIndices = distSquareMat.argsort() 
    indices = sortedIndices[:k] 
    labelCount = {} 
    for i in indices:
        label = labels[i]
        labelCount[label] = labelCount.get(label, 0) + 1

    sortedCount = sorted(labelCount.items(), key=opt.itemgetter(1), reverse=True)
    return sortedCount[0][0]

best_acc = 0
for epoch in range(1, 2):
    if best_acc == 1:
        break
    Train, Train_label, Test, Test_label = diedai(dataa)
    Train=np.array(Train)
    Test=np.array(Test)
    result = []
    for i in range(len(Test)):
        result.append(kNN(Train, Train_label, Test[i], 3))
    m = 0
    for i in range(len(result)):
        if result[i] == Test_label[i]:
            m+=1
    acc = m/len(result)
    if acc > best_acc:
        best_acc = acc
    log = 'Epoch: {:03d}, acc: {:.4f}'
    print(log.format(epoch, best_acc))