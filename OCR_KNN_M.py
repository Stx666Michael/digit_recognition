import numpy as np     #导入numpy工具包
from os import listdir #使用listdir模块，用于访问本地文件
from sklearn import neighbors
import PIL.Image as image
import random
import time

def img2vector(fileName):    
    f = open(fileName,'rb')
    data = []
    img = image.open(f)
    m,n = img.size
    for i in range(m):
        for j in range(n):
            x = img.getpixel((i,j))
            data.append(round(x/255))
    f.close()
    return np.mat(data)

def readDataSet(path):    
    fileList = listdir(path)    #获取文件夹下的所有文件 
    numFiles = len(fileList)    #统计需要读取的文件的数目
    print("Total:",numFiles)
    dataSet = np.zeros([numFiles,784],int)    #用于存放所有的数字文件
    hwLabels = np.zeros([numFiles])#用于存放对应的标签(与神经网络的不同)
    for i in range(numFiles):      #遍历所有的文件
        filePath = fileList[i]     #获取文件名称/路径   
        digit = int(filePath.split('_')[0])   #通过文件名获取标签     
        hwLabels[i] = digit        #直接存放数字，并非one-hot向量
        dataSet[i] = img2vector(path +'/'+filePath)    #读取文件内容
        if (i%1000 == 0):
            print(i)
    return dataSet,hwLabels

def readDataSet_R(path,num):    
    fileList = listdir(path)    #获取文件夹下的所有文件 
    numFiles = len(fileList)    #统计需要读取的文件的数目
    print("Ramdom:",num)
    dataSet = np.zeros([num,784],int)    #用于存放所有的数字文件
    hwLabels = np.zeros([num])#用于存放对应的标签(与神经网络的不同)
    Sample = random.sample(range(numFiles),num)
    for i in range(num):      #遍历所有的文件
        filePath = fileList[Sample[i]]     #获取文件名称/路径   
        digit = int(filePath.split('_')[0])   #通过文件名获取标签     
        hwLabels[i] = digit        #直接存放数字，并非one-hot向量
        dataSet[i] = img2vector(path +'/'+filePath)    #读取文件内容
        if (i%(num/100) == 0):
            print("\r",'█'*int(20*i/num),100*i/num+1,"%",end="")
    return dataSet,hwLabels

#read dataSet
print("Training...")
#train_dataSet, train_hwLabels = readDataSet('trainingDigits_M')
train_dataSet, train_hwLabels = readDataSet_R('trainingDigits_M',5000)

knn = neighbors.KNeighborsClassifier(algorithm='kd_tree', n_neighbors=3)
knn.fit(train_dataSet, train_hwLabels)
print("\nTraining complete.")

#read testing dataSet
print("\nTesting...")
#dataSet,hwLabels = readDataSet('testDigits_M')
dataSet,hwLabels = readDataSet_R('testDigits_M',1000)

res = knn.predict(dataSet)  #对测试集进行预测
error_num = np.sum(res != hwLabels) #统计分类错误的数目
num = len(dataSet)          #测试集的数目
print("\n\nTotal num:",num," Wrong num:", \
      error_num,"  Accuracy:",1 - error_num / float(num))
time.sleep(10)
