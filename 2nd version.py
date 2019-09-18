import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
##np.seterr(all='ignore')

#train_csv = pd.read_csv("C:\\Users\\castrojl\\Desktop\\train.csv")
#test_csv = pd.read_csv("C:\\Users\\castrojl\\Desktop\\test.csv")

#train_DF = pd.DataFrame(train_csv)
#test_DF = pd.DataFrame(test_csv)
#print(train_DF.shape)

#TrainingDataTarget = []
#TrainingDataValue = []
#TestData = []

##TrainingDataValue.append(normalize(np.array(train_DF.loc[1][1:]).reshape(784, 1)))
##print(len(TrainingDataValue))
#for i in tqdm(range(len(train_DF))):
#    if train_DF.loc[i][0] == 0:
#        target = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#    elif train_DF.loc[i][0] == 1:
#        target = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
#    elif train_DF.loc[i][0] == 2:
#        target = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
#    elif train_DF.loc[i][0] == 3:
#        target = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
#    elif train_DF.loc[i][0] == 4:
#        target = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
#    elif train_DF.loc[i][0] == 5:
#        target = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
#    elif train_DF.loc[i][0] == 6:
#        target = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
#    elif train_DF.loc[i][0] == 7:
#        target = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
#    elif train_DF.loc[i][0] == 8:
#        target = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
#    elif train_DF.loc[i][0] == 9:
#        target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
#    TrainingDataValue.append(train_DF.loc[i][1:])
#    TrainingDataTarget.append(target)
##for i in tqdm(range(len(test_DF))):
##    TestData.append(normalize(np.array(test_DF.loc[i]).reshape(784, 1)))

#np.save("TrainingDataTarget", TrainingDataTarget)
#np.save("TrainingDataValue", TrainingDataValue)

#TestData = np.load("TestData.npy", allow_pickle=True)

##plt.imshow(TrainingData[501][1].reshape(28, 28))
##print(TrainingData[501][0])
##plt.show()

##plt.imshow(TestData[501].reshape(28, 28))
##plt.show()

def stablesoftmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)

def dtanH(x):
    return 1 - np.tanh(x)**2

def tanH(x):
    return np.tanh(x)

def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

def softmax_d(z):
    return softmax_broadcast(z) * (1 - softmax_broadcast(z))

def cross_ent(y, yhat):
    return -1 * np.sum(y * np.log(yhat+1e-16))

#def cross_entropy(predictions, targets, epsilon=1e-12):
#    """
#    Computes cross entropy between targets (encoded as one-hot vectors)
#    and predictions. 
#    Input: predictions (N, k) ndarray
#           targets (N, k) ndarray        
#    Returns: scalar
#    """
#    predictions = np.clip(predictions, epsilon, 1. - epsilon)
#    N = predictions.shape[0]
#    ce = -np.sum(targets*np.log(predictions+1e-9))/N
#    return ce

def cross_ent_d(y, yhat):
    return yhat - y

def softmax_broadcast(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

w1 = np.random.randn(784, 1000)
b1 = np.random.randn(1, 1000)

w2 = np.random.randn(1000, 800)
b2 = np.random.randn(1, 800)

w3 = np.random.randn(800, 600)
b3 = np.random.randn(1, 600)

w4 = np.random.randn(600, 300)
b4 = np.random.randn(1, 300)

w5 = np.random.randn(300, 10)
b5 = np.random.randn(1, 10)

iterations = 10
lr = 0.0001
costlist = []

TrainingDataTarget = np.load("TrainingDataTarget.npy", allow_pickle=True)
TrainingDataValue = np.load("TrainingDataValue.npy", allow_pickle=True)

#plt.imshow(TrainingDataValue[1].reshape(28, 28))
#plt.show()
print(TrainingDataTarget.shape)
#print(w1, "\n")
#print(b1, "\n")

for i in tqdm(range(iterations)):

    InputData1 = TrainingDataValue
    TargetData1 = TrainingDataTarget

    z1 = np.dot(InputData1, w1) + b1
    a1 = tanH(z1)
    #a1 - 42000 x 1000  

    z2 = np.dot(a1, w2) + b2
    a2 = tanH(z2)
    #a2 - 42000 x 1000
    
    z3 = np.dot(a2, w3) + b3
    a3 = tanH(z3)
    #a3 - 42000 x 800

    z4 = np.dot(a3, w4) + b4
    a4 = tanH(z4)
    #a4 - 42000 x 500

    z5 = np.dot(a4, w5) + b5
    a5 = softmax_broadcast(z5)
    print(a5.shape)
    #a5 - 42000 x 10

    cost = cross_ent(TargetData1, a5)
    costlist.append(cost)
    print(cost)

    #backprop
    dcda5 = cross_ent_d(TargetData1, a5)
    da5dz5 = softmax_d(z5)
    dz5dw5 = a4

    dz5da4 = w5
    da4dz4 = dtanH(z4)
    dz4dw4 = a3

    dz4a3 = w4
    da3dz3 = dtanH(z3)
    dz3dw3 = a2

    dz3da2 = w3
    da2dz2 = dtanH(z2)
    dz2dw2 = a1

    dz2da1 = w2
    da1dz1 = dtanH(z1)
    dz1dw1 = InputData1

    dw5 = dcda5 * da5dz5
    db5 = np.sum(dw5, axis=0, keepdims=True)
    w5 = w5 - lr * np.dot(dz5dw5.T, dw5)
    b5 = b5 - lr * db5

    #print(w5)

    dw4 = np.dot(dw5, dz5da4.T) * da4dz4
    db4 = np.sum(dw4, axis=0, keepdims=True)
    w4 = w4 - lr * np.dot(dz4dw4.T, dw4)
    b4 = b4 - lr * db4

    #print(w4)

    dw3 = np.dot(dw4, dz4a3.T) * da3dz3
    db3 = np.sum(dw3, axis=0, keepdims=True)
    w3 = w3 - lr * np.dot(dz3dw3.T, dw3)
    b3 = b3 - lr * db3

    #print(w3)

    dw2 = np.dot(dw3, dz3da2.T) * da2dz2
    db2 = np.sum(dw2, axis=0, keepdims=True)
    w2 = w2 - lr * np.dot(dz2dw2.T, dw2)
    b2 = b2 - lr * db2

    #print(w2)

    dw1 = np.dot(dw2, dz2da1.T) * da1dz1
    db1 = np.sum(dw1, axis=0, keepdims=True)
    w1 = w1 - lr * np.dot(dz1dw1.T, dw1)
    b1 = b1 - lr * db1

#np.save("weights1", w1)
#np.save("weights2", w2)
#np.save("weights3", w3)
#np.save("weights4", w4)
#np.save("weights5", w5)
#np.save("bias1", b1)
#np.save("bias2", b2)
#np.save("bias3", b3)
#np.save("bias4", b4)
#np.save("bias5", b5)
print(costlist)
plt.plot(costlist)
plt.show()

