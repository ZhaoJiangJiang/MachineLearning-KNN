from numpy import *
import matplotlib.pyplot as plt
import operator
from os import listdir
plt.rcParams['font.sans-serif'] = ['SimHei']

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# KNN 分类算法
# 参数解释
# inX 用于分类的输入向量
# dataSet 训练样本集
# labels 标签向量
# k 选择最近邻居的数目
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()    # 返回distances值从小到大的 下标索引 序列
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  # 取第i个小的类型
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 把字典里第i个类型对应的value + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) # key:用来比较的值 operator.itemgetter(1) 表示比较第一维
    return sortedClassCount[0][0]

## 使用KNN算法改进约会网站的配对效果

# 文本处理方法
# 参数解释
# filename 文件名
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()     # strip() 方法去掉所有回车字符
        listFromLine = line.split('\t') # 把当前行通过 \t 分割成List
        returnMat[index, :] = listFromLine[0:3] # 把List的前3个 赋值给 returnMat的index行
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

# 归一化特征值
# 参数解释
# dataSet 训练样本集
def autoNorm(dataSet):
    minVals = dataSet.min(0)    # 返回矩阵中每一列的最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]    # 行数
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

# 分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix("data/datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]    # 训练集数量 的总行数
    numTestVecs = int(m * hoRatio)  # 测试集 为 训练集 的 10%
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classfier came back with : %d, the real answer is: %d"%(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is : %f"%(errorCount / float(numTestVecs)))

# 小程序
# 输入：三个特征值
# 输出：属于的类型
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix("data/datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])     # 这里就是 ffMiles percentTats iceCream 这三个变量的 输入值
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)  # 第一个参数 把 特征值归一化
    print("You will probably like this person:", resultList[classifierResult-1])


## 手写识别系统

# 将图像数据转换成List
# 32 * 32 的图像矩阵 转换成 1 * 1024 的List
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

# 手写识别系统的测试代码
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('data/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])    # 这是真实答案
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector("data/trainingDigits/%s" %fileNameStr)
    testFileList = listdir('data/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('data/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with : %d, the real answer is : %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("the total number of errors is : %d" % errorCount)
    print("the total error rate is : %f" % (errorCount/float(mTest)))
















