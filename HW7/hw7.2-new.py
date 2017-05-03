from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

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
    for j in range(0, 28):
            for k in range(0, 28):
                if randomizedImages[i][j][k] == -1:
                    oldpi[i][j][k] = 0
                    pi[i][j][k] = 0
                else:
                    oldpi[i][j][k] = 1
                    pi[i][j][k] = 1

c = [-.8, -.4, 0, .4, .8]
theta2 = 2
oldpiCount = 784
minAcc = 1000
maxAcc = -1000
totalAcc = 0
maxIndex = -1
minIndex = -1

true_positives = np.zeros(5)
false_positives = np.zeros(5)
total_positives = np.zeros(5)
total_negatives = np.zeros(5)

for l in range (0, 5):
    for i in range(0, 500):
        oldpi[i] = [[0 for x in range(28)] for y in range(28)]
        pi[i] = [[0 for x in range(28)] for y in range(28)]
        for j in range(0, 28):
                for k in range(0, 28):
                    if randomizedImages[i][j][k] == -1:
                        oldpi[i][j][k] = 0
                        pi[i][j][k] = 0
                    else:
                        oldpi[i][j][k] = 1
                        pi[i][j][k] = 1

    for k in range (0, 500):
        oldpiCount = 784
        print k
        count = 0
        while(oldpiCount != 0 and count < 100):
            oldpiCount = 0
            count += 1
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

                    mine_num = theta2 * randomizedImages[k][i][j]
                    mine_denom1 = (-1 * theta2) * randomizedImages[k][i][j]
                    mine_denom2 = mine_num

                    if j + 1 < 28:
                        up_num = c[l] * (2 * oldpi[k][i][j+1] - 1) 
                        up_denom1 = (-1 * c[l]) * (2 * oldpi[k][i][j+1] - 1) 
                        up_denom2 = up_num
                    if j - 1 >= 0:
                        down_num = c[l] * (2 * oldpi[k][i][j-1] - 1) 
                        down_denom1 = (-1 * c[l]) * (2 * oldpi[k][i][j-1] - 1) 
                        down_denom2 = down_num
                    if i + 1 < 28:
                        right_num = c[l] * (2 * oldpi[k][i+1][j])
                        right_denom1 = (-1 * c[l]) * (2 * oldpi[k][i+1][j] - 1)
                        right_denom2 = right_num

                    if i - 1 < 28:
                        left_num = c[l] * (2 * oldpi[k][i-1][j] - 1)
                        left_denom1 = (-1 * c[l]) * (2 * oldpi[k][i-1][j] - 1)
                        left_denom2 = left_num
                    oldpi[k][i][j] = pi[k][i][j]
                    pi[k][i][j] = (np.exp(mine_num + up_num + down_num + right_num + left_num))/(np.exp(mine_denom1 + up_denom1 + down_denom1 + right_denom1 + left_denom1) + np.exp(mine_denom2 + up_denom2 + down_denom2 + right_denom2 + left_denom2))
                    
                    if abs(pi[k][i][j] - oldpi[k][i][j]) > .0000000001:
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

        for i in range(0, 28):
            for j in range(0, 28):
                if normalImages[k][i][j] == 1:
                    total_positives[l] += 1
                    if updatedImages[k][i][j] == 1:
                        true_positives[l] += 1
                if normalImages[k][i][j] == -1:
                    total_negatives[l] += 1
                    if updatedImages[k][i][j] == 1:
                        false_positives[l] += 1

true_positives /= total_positives
false_positives /= total_negatives

plt.figure()
plt.scatter(false_positives, true_positives, s=80, facecolors='none', edgecolors='r')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('roc-curve.png', bbox_inches='tight')

print(true_positives)
print(false_positives)