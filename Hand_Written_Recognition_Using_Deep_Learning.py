import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from sklearn.preprocessing import normalize

#np.seterr(all='ignore')

#train_csv = pd.read_csv("C:\\Users\\castrojl\\Desktop\\train.csv")
#test_csv = pd.read_csv("C:\\Users\\castrojl\\Desktop\\test.csv")

#train_DF = pd.DataFrame(train_csv)
#test_DF = pd.DataFrame(test_csv)
#TrainingData = []
#TestData = []
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
#    TrainingData.append((np.array(target), np.array(train_DF.loc[i][1:])))

#for i in tqdm(range(len(test_DF))):
#    TestData.append(np.array(test_DF.loc[i]))

#np.save("TrainingData", TrainingData)
#np.save("TestData", TestData)

TrainingData = np.load("TrainingData.npy", allow_pickle=True)
TestData = np.load("TestData.npy", allow_pickle=True)

#plt.imshow(TrainingData[501][1].reshape(28, 28))
#print(TrainingData[501][0])
#plt.show()

#plt.imshow(TestData[501].reshape(28, 28))
#plt.show()

def stablesoftmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)

def tanh_p(x):
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
    return stablesoftmax(z) * (1 - stablesoftmax(z))

def cross_ent(y, yhat):
    return -1 * np.sum(y * np.log(yhat))

def cross_ent_d(y, yhat):
    return yhat - y

w1 = np.random.randn(1200, 784)
b1 = np.random.randn(1200, 1)

w2 = np.random.randn(800, 1200)
b2 = np.random.randn(800, 1)

w3 = np.random.randn(600, 800)
b3 = np.random.randn(600, 1)

w4 = np.random.randn(300, 600)
b4 = np.random.randn(300, 1)

w5 = np.random.randn(10, 300)
b5 = np.random.randn(10, 1)

iterations = 10000
lr = 0.1
costlist = []

for i in tqdm(range(iterations)):
    try:
        random = np.random.choice(len(TrainingData))
        InputData1 = normalize(TrainingData[random][1].reshape(784, 1))
        TargetData1 = normalize(TrainingData[random][0].reshape(10, 1))

        #plt.imshow(InputData1.reshape(28, 28))
        #plt.show()

        z1 = np.dot(w1, InputData1) + b1
        a1 = tanH(z1)

        z2 = np.dot(w2, a1) + b2
        a2 = tanH(z2)

        z3 = np.dot(w3, a2) + b3
        a3 = tanH(z3)

        z4 = np.dot(w4, a3) + b4
        a4 = tanH(z4)

        z5 = np.dot(w5, a4) + b5
        a5 = stablesoftmax(z5)

        if i % 2000 == 0:
            c = 0
            for x in range(len(TrainingData)):

                InputData2 = normalize(TrainingData[x][1].reshape(784, 1))
                TargetData2 = normalize(TrainingData[x][0].reshape(10, 1))

                z1 = np.dot(w1, InputData2) + b1
                a1 = tanH(z1)

                z2 = np.dot(w2, a1) + b2
                a2 = tanH(z2)

                z3 = np.dot(w3, a2) + b3
                a3 = tanH(z3)

                z4 = np.dot(w4, a3) + b4
                a4 = tanH(z4)

                z5 = np.dot(w5, a4) + b5
                a5 = stablesoftmax(z5)

                c += cross_ent(TargetData2, a5)
            print(c)
            costlist.append(c)

        #print(cost)

        #backprop
        dcda5 = cross_ent_d(TargetData1, a5)
        da5dz5 = softmax_d(z5)
        dz5dw5 = a4

        dz5da4 = w5
        da4dz4 = tanh_p(z4)
        dz4dw4 = a3

        dz4a3 = w4
        da3dz3 = tanh_p(z3)
        dz3dw3 = a2

        dz3da2 = w3
        da2dz2 = tanh_p(z2)
        dz2dw2 = a1

        dz2da1 = w2
        da1dz1 = tanh_p(z1)
        dz1dw1 = InputData1

        dw5 = dcda5 * da5dz5
        db5 = np.sum(dw5, axis=1, keepdims=True)
        w5 = w5 - lr * np.dot(dw5, dz5dw5.T)
        b5 = b5 - lr * db5

        #print(w5)

        dw4 = np.dot(dz5da4.T, dw5) * da4dz4
        db4 = np.sum(dw4, axis=1, keepdims=True)
        w4 = w4 - lr * np.dot(dw4, dz4dw4.T)
        b4 = b4 - lr * db4

        #print(w4)

        dw3 = np.dot(dz4a3.T, dw4) * da3dz3
        db3 = np.sum(dw3, axis=1, keepdims=True)
        w3 = w3 - lr * np.dot(dw3, dz3dw3.T)
        b3 = b3 - lr * db3

        #print(w3)

        dw2 = np.dot(dz3da2.T, dw3) * da2dz2
        db2 = np.sum(dw2, axis=1, keepdims=True)
        w2 = w2 - lr * np.dot(dw2, dz2dw2.T)
        b2 = b2 - lr * db2

        #print(w2)

        dw1 = np.dot(dz2da1.T, dw2) * da1dz1
        db1 = np.sum(dw1, axis=1, keepdims=True)
        w1 = w1 - lr * np.dot(dw1, dz1dw1.T)
        b1 = b1 - lr * db1

        #print(w1)
    except:
        print(a1)
        print(a2)
        print(a3)
        print(a4)
        print(a5)
        print(w1)
        print(w2)
        print(w3)
        print(w4)
        print(w5)
        print(b1)
        print(b2)
        print(b3)
        print(b4)
        print(b5)

np.save("weights1", w1)
np.save("weights2", w2)
np.save("weights3", w3)
np.save("weights4", w4)
np.save("weights5", w5)
np.save("bias1", b1)
np.save("bias2", b2)
np.save("bias3", b3)
np.save("bias4", b4)
np.save("bias5", b5)

print(costlist)
plt.plot(costlist)
plt.show()

