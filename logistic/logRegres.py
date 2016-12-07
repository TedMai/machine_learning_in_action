# -*- coding: utf-8 -*-

from math import exp
from numpy import *

def loadDataSet():
    dataMat=[]; labelMat=[]
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0/(1+exp(-inX))


def gradAscent(dataMatIn, classLabels):
    """
    梯度上升
    梯度上升
    :param dataMatIn:
    :param classLabels:
    :return:
    """
    dataMatrix = mat(dataMatIn) # samples by features + 1
    labelMat = mat(classLabels).transpose() # 转为列向量
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights) # samples by 1 向量
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


def plotBestFit(weightsMat):
    import matplotlib.pyplot as plt
    weights = weightsMat.getA()
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]

    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    # 画出最佳拟合直线
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0] - weights[1]*x) / weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stocGradAscent0(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i) + 0.01 # alpha 随着迭代次数的增多而减小，但永远不会到达0
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights += alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1
    else:
        return 0


def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabel = []
    for line in frTrain.readlines():
        currentLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currentLine[i]))
        trainingSet.append(lineArr)
        trainingLabel.append(float(currentLine[21]))

    trainWeights = stocGradAscent0(array(trainingSet), trainingLabel, 500)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currentLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currentLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currentLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)) / numTestVec
    print 'the error rate of this test is: %f' % errorRate
    return errorRate

def colicTest2():
    """
        使用scikit-learn 工具包
    :return:
    """
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabel = []
    for line in frTrain.readlines():
        currentLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currentLine[i]))
        trainingSet.append(lineArr)
        trainingLabel.append(float(currentLine[21]))
    from sklearn import linear_model
    sgdc = linear_model.SGDClassifier()
    sgdc.fit(array(trainingSet), array(trainingLabel))
    testingSet = []; testingLabel = []
    for line in frTest.readlines():
        currentLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currentLine[i]))
        testingSet.append(lineArr)
        testingLabel.append(float(currentLine[21]))
    sgdc_predict = sgdc.predict(testingSet)
    errorCount = 0
    for i in range(len(testingLabel)):
        if int(sgdc_predict[i]) != int(testingLabel[i]):
            errorCount += 1
    error_rate =  float(errorCount) / len(testingLabel)
    print error_rate
    return error_rate


def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest2()
    print "after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests))