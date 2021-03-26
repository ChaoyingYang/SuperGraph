import numpy as np
import scipy.io as sio
import math
import time
from scipy.fftpack import fft, fftshift, ifft
from scipy.fftpack import fftfreq
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy
from matplotlib import cm

start = time.time()
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

Start=time.time()    
sample_num=10
G1=19
#----------------------crack 0-------------------
mj=0
temp00=[]
infile=r'Gearbox\Dataset_1\LW-00\1500-0.txt'
temp1=loadDatadet(infile,k)
temp1=temp1[4001+sample_num*400*G1:4002+sample_num*400*(G1+1)]
for i in range(0,len(temp1)):
    temp00.append(float(temp1[i][k]))
temp00=normalization(temp00)
for j in range(0,400*sample_num,400):
    dataset.append(temp00[0+j:400+j])



temp01=[] 
infile=r'Gearbox\Dataset_1\LW-00\1500-2.txt'
temp1=loadDatadet(infile,k)
temp1=temp1[4001+sample_num*400*G1:4002+sample_num*400*(G1+1)]
for i in range(0,len(temp1)):
    temp01.append(float(temp1[i][k]))
temp01=normalization(temp01)
for j in range(0,400*sample_num,400):
    dataset.append(temp01[0+j:400+j])



infile=r'Gearbox\Dataset_1\LW-00\1500-4.txt'
temp02=[]
temp1=loadDatadet(infile,k)
temp1=temp1[4001+sample_num*400*G1:4002+sample_num*400*(G1+1)]
for i in range(0,len(temp1)):
    temp02.append(float(temp1[i][k]))
temp02=normalization(temp02)
for j in range(0,400*sample_num,400):
    dataset.append(temp02[0+j:400+j])

    
infile=r'Gearbox\Dataset_1\LW-00\1500-6.txt'
temp03=[]
temp1=loadDatadet(infile,k)
temp1=temp1[4001+sample_num*400*G1:4002+sample_num*400*(G1+1)]
for i in range(0,len(temp1)):
    temp03.append(float(temp1[i][k]))
temp03=normalization(temp03)
for j in range(0,400*sample_num,400):
    dataset.append(temp03[0+j:400+j])


infile=r'Gearbox\Dataset_1\LW-00\1500-8.txt'
temp04=[]
temp1=loadDatadet(infile,k)
temp1=temp1[4001+sample_num*400*G1:4002+sample_num*400*(G1+1)]
for i in range(0,len(temp1)):
    temp04.append(float(temp1[i][k]))
temp04=normalization(temp04)
for j in range(0,400*sample_num,400):
    dataset.append(temp04[0+j:400+j])

    
infile=r'Gearbox\Dataset_1\LW-00\1500-10.txt'
temp05=[]
temp1=loadDatadet(infile,k)
temp1=temp1[4001+sample_num*400*G1:4002+sample_num*400*(G1+1)]
for i in range(0,len(temp1)):
    temp05.append(float(temp1[i][k]))
temp05=normalization(temp05)
for j in range(0,400*sample_num,400):
    dataset.append(temp05[0+j:400+j])

     
# #----------------------crack 5-------------------
mj=1
temp10=[]
infile=r'Gearbox\Dataset_1\LW-01\1500-0.txt'
temp1=loadDatadet(infile,k)
temp1=temp1[4001+sample_num*400*G1:4002+sample_num*400*(G1+1)]
for i in range(0,len(temp1)):
    temp10.append(float(temp1[i][k]))
temp10=normalization(temp10)
for j in range(0,400*sample_num,400):
    dataset.append(temp10[0+j:400+j])


temp11=[] 
infile=r'Gearbox\Dataset_1\LW-01\1500-2.txt'
temp1=loadDatadet(infile,k)
temp1=temp1[4001+sample_num*400*G1:4002+sample_num*400*(G1+1)]
for i in range(0,len(temp1)):
    temp11.append(float(temp1[i][k]))
temp11=normalization(temp11)
for j in range(0,400*sample_num,400):
    dataset.append(temp11[0+j:400+j])



infile=r'Gearbox\Dataset_1\LW-01\1500-4.txt'
temp12=[]
temp1=loadDatadet(infile,k)
temp1=temp1[4001+sample_num*400*G1:4002+sample_num*400*(G1+1)]
for i in range(0,len(temp1)):
    temp12.append(float(temp1[i][k]))
temp12=normalization(temp12)
for j in range(0,400*sample_num,400):
    dataset.append(temp12[0+j:400+j])

    
infile=r'Gearbox\Dataset_1\LW-01\1500-6.txt'
temp13=[]
temp1=loadDatadet(infile,k)
temp1=temp1[4001+sample_num*400*G1:4002+sample_num*400*(G1+1)]
for i in range(0,len(temp1)):
    temp13.append(float(temp1[i][k]))
temp13=normalization(temp13)
for j in range(0,400*sample_num,400):
    dataset.append(temp13[0+j:400+j])



infile=r'Gearbox\Dataset_1\LW-01\1500-8.txt'
temp14=[]
temp1=loadDatadet(infile,k)
temp1=temp1[4001+sample_num*400*G1:4002+sample_num*400*(G1+1)]
for i in range(0,len(temp1)):
    temp14.append(float(temp1[i][k]))
temp14=normalization(temp14)
for j in range(0,400*sample_num,400):
    dataset.append(temp14[0+j:400+j])

    
infile=r'Gearbox\Dataset_1\LW-01\1500-10.txt'
temp15=[]
temp1=loadDatadet(infile,k)
temp1=temp1[4001+sample_num*400*G1:4002+sample_num*400*(G1+1)]
for i in range(0,len(temp1)):
    temp15.append(float(temp1[i][k]))
temp15=normalization(temp15)
for j in range(0,400*sample_num,400):
    dataset.append(temp15[0+j:400+j])

    
# #----------------------crack 10-------------------
mj=2
temp20=[]
infile=r'Gearbox\Dataset_1\LW-02\1500-0.txt'
temp1=loadDatadet(infile,k)
temp1=temp1[4001+sample_num*400*G1:4002+sample_num*400*(G1+1)]
for i in range(0,len(temp1)):
    temp20.append(float(temp1[i][k]))
temp20=normalization(temp20)
for j in range(0,400*sample_num,400):
    dataset.append(temp20[0+j:400+j])
 

temp21=[] 
infile=r'Gearbox\Dataset_1\LW-02\1500-2.txt'
temp1=loadDatadet(infile,k)
temp1=temp1[4001+sample_num*400*G1:4002+sample_num*400*(G1+1)]
for i in range(0,len(temp1)):
    temp21.append(float(temp1[i][k]))
temp21=normalization(temp21)
for j in range(0,400*sample_num,400):
    dataset.append(temp21[0+j:400+j])



infile=r'Gearbox\Dataset_1\LW-02\1500-4.txt'
temp22=[]
temp1=loadDatadet(infile,k)
temp1=temp1[4001+sample_num*400*G1:4002+sample_num*400*(G1+1)]
for i in range(0,len(temp1)):
    temp22.append(float(temp1[i][k]))
temp22=normalization(temp22)
for j in range(0,400*sample_num,400):
    dataset.append(temp22[0+j:400+j])

    
infile=r'Gearbox\Dataset_1\LW-02\1500-6.txt'
temp23=[]
temp1=loadDatadet(infile,k)
temp1=temp1[4001+sample_num*400*G1:4002+sample_num*400*(G1+1)]
for i in range(0,len(temp1)):
    temp23.append(float(temp1[i][k]))
temp23=normalization(temp23)
for j in range(0,400*sample_num,400):
    dataset.append(temp23[0+j:400+j])


infile=r'Gearbox\Dataset_1\LW-02\1500-8.txt'
temp24=[]
temp1=loadDatadet(infile,k)
temp1=temp1[4001+sample_num*400*G1:4002+sample_num*400*(G1+1)]
for i in range(0,len(temp1)):
    temp24.append(float(temp1[i][k]))
temp24=normalization(temp24)
for j in range(0,400*sample_num,400):
    dataset.append(temp24[0+j:400+j])

    
infile=r'Gearbox\Dataset_1\LW-02\1500-10.txt'
temp25=[]
temp1=loadDatadet(infile,k)
temp1=temp1[4001+sample_num*400*G1:4002+sample_num*400*(G1+1)]
for i in range(0,len(temp1)):
    temp25.append(float(temp1[i][k]))
temp25=normalization(temp25)
for j in range(0,400*sample_num,400):
    dataset.append(temp25[0+j:400+j])

    
# #----------------------crack 15-------------------
mj=3
temp30=[]
infile=r'Gearbox\Dataset_1\LW-03\1500-0.txt'
temp1=loadDatadet(infile,k)
temp1=temp1[4001+sample_num*400*G1:4002+sample_num*400*(G1+1)]
for i in range(0,len(temp1)):
    temp30.append(float(temp1[i][k]))
temp30=normalization(temp30)
for j in range(0,400*sample_num,400):
    dataset.append(temp30[0+j:400+j])


temp31=[] 
infile=r'Gearbox\Dataset_1\LW-03\1500-2.txt'
temp1=loadDatadet(infile,k)
temp1=temp1[4001+sample_num*400*G1:4002+sample_num*400*(G1+1)]
for i in range(0,len(temp1)):
    temp31.append(float(temp1[i][k]))
temp31=normalization(temp31)
for j in range(0,400*sample_num,400):
    dataset.append(temp31[0+j:400+j])



infile=r'Gearbox\Dataset_1\LW-03\1500-4.txt'
temp32=[]
temp1=loadDatadet(infile,k)
temp1=temp1[4001+sample_num*400*G1:4002+sample_num*400*(G1+1)]
for i in range(0,len(temp1)):
    temp32.append(float(temp1[i][k]))
temp32=normalization(temp32)
for j in range(0,400*sample_num,400):
    dataset.append(temp32[0+j:400+j])

    
infile=r'Gearbox\Dataset_1\LW-03\1500-6.txt'
temp33=[]
temp1=loadDatadet(infile,k)
temp1=temp1[4001+sample_num*400*G1:4002+sample_num*400*(G1+1)]
for i in range(0,len(temp1)):
    temp33.append(float(temp1[i][k]))
temp33=normalization(temp33)
for j in range(0,400*sample_num,400):
    dataset.append(temp33[0+j:400+j])


infile=r'Gearbox\Dataset_1\LW-03\1500-8.txt'
temp34=[]
temp1=loadDatadet(infile,k)
temp1=temp1[4001+sample_num*400*G1:4002+sample_num*400*(G1+1)]
for i in range(0,len(temp1)):
    temp34.append(float(temp1[i][k]))
temp34=normalization(temp34)
for j in range(0,400*sample_num,400):
    dataset.append(temp34[0+j:400+j])

    
infile=r'Gearbox\Dataset_1\LW-03\1500-10.txt'
temp35=[]
temp1=loadDatadet(infile,k)
temp1=temp1[4001+sample_num*400*G1:4002+sample_num*400*(G1+1)]
for i in range(0,len(temp1)):
    temp35.append(float(temp1[i][k]))
temp35=normalization(temp35)
for j in range(0,400*sample_num,400):
    dataset.append(temp35[0+j:400+j])
    
y=[]

for i in range(60):
    y.append(0)

for i in range(60):
    y.append(1)
    
for i in range(60):
    y.append(2)
    
for i in range(60):
    y.append(3)

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
        if m<0.3*len(arr):
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
for epoch in range(1, 11):
    if best_acc == 1:
        break
    Train, Train_label, Test, Test_label = diedai(dataset)
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


