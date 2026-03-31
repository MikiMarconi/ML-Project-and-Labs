import numpy as np
import sys

def fillMatrix(m, n):
    arrayValues = np.zeros((m, n), dtype = np.float64)

    for i in range(m):
        for j in range(n):
            arrayValues[i][j] = i*j
    
    return arrayValues

def normalizeCol(arrayVal):

    arrayRet = np.zeros((3,4), dtype = np.float32)

    sumColumn = arrayVal.sum(axis = 0)

    for j in range(arrayVal.shape[0]):
        for i in range(arrayVal.shape[1]):
            arrayRet[j][i] = arrayVal[j][i]/sumColumn[i]

    return arrayRet


def normalizeRaw(arrayVal):

    arrayRet = np.zeros((4,3), dtype = np.float32)

    sumColumn = arrayVal.sum(axis = 1)

    for j in range(arrayVal.shape[0]):
        for i in range(arrayVal.shape[1]):
            arrayRet[j][i] = arrayVal[j][i]/sumColumn[j]
    return arrayRet


def normalizeZero(arrayRand):
    arrayRet = np.zeros((arrayRand.shape[0], arrayRand.shape[1]), dtype = np.float32)
    
    for i in range(arrayRand.shape[0]):
        for j in range(arrayRand.shape[1]):
            if arrayRand[i][j] > 0:
                arrayRet[i][j] = arrayRand[i][j]

    return arrayRet

def sumProdMatrix(matrix1, matrix2):
    sum = 0
    matrixProd = np.dot(matrix1, matrix2)
    sum = matrixProd.sum()
    return sum

m = int(sys.argv[1])
n = int(sys.argv[2])

arrayVal = np.array([[1.0, 2.0, 6.0, 4.0],
                     [3.0, 4.0, 3.0, 7.0],
                     [1.0, 4.0, 6.0, 9.0]])

arrayValRaw = np.array([[1.0, 3.0, 1.0],
                    [2.0, 4.0, 4.0],
                    [6.0, 3.0, 6.0],
                    [4.0, 7.0, 9.0]])

arrayRand = np.array([[1.0, 3.0, -1.0],
                     [2.0, -4.0, 4.0],
                     [-6.0, 3.0, 6.0],
                     [-4.0, 7.0, -9.0]])

matrix1 = np.array([[1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0]])

matrix2 = np.array([[1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0]])

print(fillMatrix(m, n))
print(normalizeCol(arrayVal))
print(normalizeRaw(arrayValRaw))
print(normalizeZero(arrayRand))
print(sumProdMatrix(matrix1, matrix2))



