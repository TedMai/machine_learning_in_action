# -*- coding: utf-8 -*-

from numpy import *

def loadSimpData():
    datMat = matrix([[1,2.1],
                     [2,1.1],
                     [1.3,1],
                     [1,1],
                     [2,1]])
    classLabels = [1.0,1.0,-1.0,-1.0,1.0]
    return  datMat, classLabels


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    """
    相当于使用平行于坐标轴的直线去截数据集，指定变化步长，直到找到最小错误率的坐标
    :param dataArr:
    :param classLabels:
    :param D:  数据权重
    :return:
    """
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0 # 确定查找次数
    bestStump={} # 保存最佳分隔
    bestClassEst = mat(zeros((m,1))) # 保存最佳预测向量
    minError = inf
    for i in range(n):  # 对每个特征进行循环
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max() # 该列的最小最大值
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps)+1):
            for inequal in ['lt','gt']: # 因为你根本不知道数据集中到底大于阙值的是正类还是小于阙值的是正类，所以只能来回变化确定
                threshVal = (rangeMin + float(j)*stepSize)  # 基准线坐标
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m,1)))
                errArr[[predictedVals == labelMat]] = 0 # 为方便计算错误率，转为0,1计算
                # 计算加权错误率
                weightedError = D.T * errArr
                # print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i,threshVal,inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
                    bestClassEst = predictedVals.copy()
    return bestStump, minError, bestClassEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = [] # 弱分类器集合
    m = shape(dataArr)[0]
    D = mat(ones((m,1)) / m) # 初始化时，所有数据赋有相同权重
    aggClassEst = mat(zeros((m,1))) # 存放最终的结果，即分类器的线性组合
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        # print "D:", D.T
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        # print "classEst: ", classEst.T
        # 以下三步 更新权重
        expon = multiply(-1*alpha*mat(classLabels).T, classEst) # multiply 为点乘的意思，这里通过引入类别标签合并权重更新的操作
        D = multiply(D, exp(expon))
        D = D / D.sum()

        aggClassEst += alpha * classEst
        # print "aggClassEst: ", aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m,1))) # 最终错误率
        errorRate = aggErrors.sum() / m
        print "total error: ", errorRate,"\n"
        if errorRate == 0.0:
            break
    return weakClassArr


def adaClassify(datToClass, classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        print aggClassEst
    return sign(aggClassEst)



