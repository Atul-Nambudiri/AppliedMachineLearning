from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

mndata = MNIST('samples')

images, labels = mndata.load_training()

normalImages = [None]*500
randomizedImages = [None]*500
updatedImages = [None]*500
oldpi = [None]*500
pi = [None]*500
fpr = [None]*5
tpr = [None]*5

for j in range (0, 500):
    for k in range (0, len(images[j])):
        if images[j][k] < .5:
            images[j][k] = -1
        else:
            images[j][k] = 1
    img = [images[j][i:i+28] for i in xrange(0,len(images[j]), 28)]
    normalImages[j] = np.array(img)
    randomizedImages[j] = np.array(img)
    updatedImages[j] = np.array(img)
    oldpi[j] = np.array(img)
    pi[j] = np.array(img)

    for i in range (0, 15):
        n = np.random.randint(0, high=784)
        row = n/28
        col = n % 28
        if randomizedImages[j][row][col] == -1:
            randomizedImages[j][row][col] = 1
        else:
            randomizedImages[j][row][col] = -1

for i in range(0, 500):
    oldpi[i] = [[0 for x in range(28)] for y in range(28)]
    pi[i] = [[0 for x in range(28)] for y in range(28)]
    pi[i] = np.array(pi[i])
    oldpi[i] = np.array(oldpi[i])
    for j in range(0, 28):
            for k in range(0, 28):
                if randomizedImages[i][j][k] == -1:
                    pi[i][j][k] = 0
                else:
                    pi[i][j][k] = 1

theta1 = .2
c = [-.8, -.4, .2, .5, .8]

theta2 = 2
oldpiCount = 784
minAcc = 1000
maxAcc = -1000
totalAcc = 0
maxIndex = -1
minIndex = -1

true_positives = np.zeros((5, 500))
false_positives = np.zeros((5, 500))

for l in range (0, 5):
    theta1 = c[l]
    for k in range (0, 500):
        oldpiCount = 784
        if k % 100 == 0:
            print(k)
        while(oldpiCount != 0):
            oldpiCount = 0
            for i in range(0, 28):
                for j in range(0, 28):
                    up_num = 0
                    down_num = 0
                    right_num = 0
                    left_num = 0
                    up_denom1 = 0
                    down_denom1 = 0
                    right_denom1 = 0
                    left_denom1 = 0
                    up_denom2 = 0
                    down_denom2 = 0
                    right_denom2 = 0
                    left_denom2 = 0

                    mine_num = theta1 * (2 * oldpi[k][i][j] - 1) + theta2 * randomizedImages[k][i][j]
                    mine_denom1 = (-1 * theta1) * (2 * oldpi[k][i][j] - 1) + (-1 * theta2) * randomizedImages[k][i][j]
                    mine_denom2 = mine_num

                    if j + 1 < 28:
                        up_num = theta1 * (2 * oldpi[k][i][j+1] - 1) + theta2 * randomizedImages[k][i][j+1]
                        up_denom1 = (-1 * theta1) * (2 * oldpi[k][i][j+1] - 1) + (-1 * theta2) * randomizedImages[k][i][j+1]
                        up_denom2 = up_num
                    if j - 1 >= 0:
                        down_num = theta1 * (2 * oldpi[k][i][j-1] - 1) + theta2 * randomizedImages[k][i][j-1]
                        down_denom1 = (-1 * theta1) * (2 * oldpi[k][i][j-1] - 1) + (-1 * theta2) * randomizedImages[k][i][j-1]
                        down_denom2 = down_num
                    if i + 1 < 28:
                        right_num = theta1 * (2 * oldpi[k][i+1][j]) + theta2 * randomizedImages[0][i+1][j]
                        right_denom1 = (-1 * theta1) * (2 * oldpi[k][i+1][j] - 1) + (-1 * theta2) * randomizedImages[k][i+1][j]
                        right_denom2 = right_num

                    if i - 1 < 28:
                        left_num = theta1 * (2 * oldpi[k][i-1][j] - 1) + theta2 * randomizedImages[k][i-1][j]
                        left_denom1 = (-1 * theta1) * (2 * oldpi[k][i-1][j] - 1) + (-1 * theta2) * randomizedImages[k][i-1][j]
                        left_denom2 = left_num
                    oldpi[k][i][j] = pi[k][i][j]
                    pi[k][i][j] = (np.exp(mine_num + up_num + down_num + right_num + left_num))/(np.exp(mine_denom1 + up_denom1 + down_denom1 + right_denom1 + left_denom1) + np.exp(mine_denom2 + up_denom2 + down_denom2 + right_denom2 + left_denom2))
                    
                    if abs(pi[k][i][j] - oldpi[k][i][j]) > .01:
                        oldpiCount = oldpiCount + 1
        currAcc = 0
        for i in range(0, 28):
            for j in range(0, 28):
                if(pi[k][i][j] >= .5):
                    updatedImages[k][i][j] = 1
                else:
                    updatedImages[k][i][j] = -1
                if updatedImages[k][i][j] == normalImages[k][i][j]:
                    currAcc += 1
        if currAcc > maxAcc:
            maxAcc = currAcc
            maxIndex = k
        if currAcc < minAcc:
            minAcc = currAcc
            minIndex = k
        totalAcc = totalAcc + currAcc

        positives = 0
        negatives = 0

        for i in range(0, 28):
            for j in range(0, 28):
                if normalImages[k][i][j] == 1:
                    positives += 1
                    if updatedImages[k][i][j] == 1:
                        true_positives[l][k] += 1
                if normalImages[k][i][j] == -1:
                    negatives += 1
                    if updatedImages[k][i][j] == 1:
                        false_positives[l][k] += 1
        true_positives[l] /= positives
        false_positives[l] /= negatives

plt.figure()
plt.plot(true_positives[0], false_positives[0])
plt.plot(true_positives[1], false_positives[1])
plt.plot(true_positives[2], false_positives[2])
plt.plot(true_positives[3], false_positives[3])
plt.plot(true_positives[4], false_positives[4])
plt.show()


