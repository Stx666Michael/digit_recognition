import numpy as np     #导入numpy工具包
from os import listdir #使用listdir模块，用于访问本地文件
from sklearn.neural_network import MLPClassifier
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

def readDataSet_R(path,num):    
    fileList = listdir(path)    #获取文件夹下的所有文件 
    numFiles = len(fileList)    #统计需要读取的文件的数目
    print("Ramdom:",num)
    dataSet = np.zeros([num,784],int)    #用于存放所有的数字文件
    hwLabels = np.zeros([num,10])#用于存放对应的标签
    Sample = random.sample(range(numFiles),num)
    for i in range(num):      #遍历所有的文件
        filePath = fileList[Sample[i]]     #获取文件名称/路径   
        digit = int(filePath.split('_')[0])   #通过文件名获取标签     
        hwLabels[i][digit] = 1.0        #将对应的one-hot标签置1
        dataSet[i] = img2vector(path +'/'+filePath)    #读取文件内容
        if (i%(num/100) == 0):
            print("\rLoading:",'█'*int(20*i/num),100*i/num+1,"%",end="")
    return dataSet,hwLabels

#read dataSet
print("Training...")
#train_dataSet, train_hwLabels = readDataSet('trainingDigits_M')
train_dataSet, train_hwLabels = readDataSet_R('trainingDigits_M',10000)

clf = MLPClassifier(solver='sgd',activation='relu',alpha=1e-4,hidden_layer_sizes=(50,50),random_state=1,max_iter=10,verbose=10,learning_rate_init=.1)
print()
print(clf)
clf.fit(train_dataSet,train_hwLabels)
print("Training complete.")

#read  testing dataSet
print("\nTesting...")
#dataSet,hwLabels = readDataSet('testDigits_M')
dataSet,hwLabels = readDataSet_R('testDigits_M',1000)

res = clf.predict(dataSet)   #对测试集进行预测
error_num = 0                #统计预测错误的数目
num = len(dataSet)           #测试集的数目
for i in range(num):         #遍历预测结果
    #比较长度为10的数组，返回包含01的数组，0为不同，1为相同
    #若预测结果与真实结果相同，则10个数字全为1，否则不全为1
    if np.sum(res[i] == hwLabels[i]) < 10: 
        error_num += 1                     
print("\nTotal num:",num," Wrong num:", \
      error_num,"  Accuracy:",1 - error_num / float(num))
time.sleep(100)
