from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np

mndata = MNIST('samples')

images, labels = mndata.load_training()

normalImages = [None]*500
randomizedImages = [None]*500
updatedImages = [None]*500
oldpi = [None]*500
pi = [None]*500

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
                    pi[i][j][k] = 0
                else:
                    pi[i][j][k] = 1

theta1 = .2
theta2 = 2
oldpiCount = 784
minAcc = 1000
maxAcc = -1000
totalAcc = 0
maxIndex = -1
minIndex = -1


for k in range (0, 500):
    oldpiCount = 784
    count = 0
    print k
    while(oldpiCount != 0 and count < 300):
        oldpiCount = 0
        count = count + 1
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
                    up_num = theta1 * (2 * oldpi[k][i][j+1] - 1)
                    up_denom1 = (-1 * theta1) * (2 * oldpi[k][i][j+1] - 1) 
                    up_denom2 = up_num
                if j - 1 >= 0:
                    down_num = theta1 * (2 * oldpi[k][i][j-1] - 1)
                    down_denom1 = (-1 * theta1) * (2 * oldpi[k][i][j-1] - 1)
                    down_denom2 = down_num
                if i + 1 < 28:
                    right_num = theta1 * (2 * oldpi[k][i+1][j])
                    right_denom1 = (-1 * theta1) * (2 * oldpi[k][i+1][j] - 1)
                    right_denom2 = right_num

                if i - 1 < 28:
                    left_num = theta1 * (2 * oldpi[k][i-1][j] - 1)
                    left_denom1 = (-1 * theta1) * (2 * oldpi[k][i-1][j] - 1) 
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

plt.figure(1)
plt.imshow(normalImages[maxIndex])
plt.savefig('most-original.png', bbox_inches='tight')
plt.figure(2)
plt.imshow(randomizedImages[maxIndex])
plt.savefig('most-noisy.png', bbox_inches='tight')
plt.figure(3)
plt.imshow(updatedImages[maxIndex])
plt.savefig('most-reconstructed.png', bbox_inches='tight')
plt.figure(4)
plt.imshow(normalImages[minIndex])
plt.savefig('least-original.png', bbox_inches='tight')
plt.figure(5)
plt.imshow(randomizedImages[minIndex])
plt.savefig('least-noisy.png', bbox_inches='tight')
plt.figure(6)
plt.imshow(updatedImages[minIndex])
plt.savefig('least-reconstructed.png', bbox_inches='tight')

print("maxAcc ", maxAcc/784.0)
print("minAcc ", minAcc/784.0)
print("totalAcc ", totalAcc/392000.0)

print("Max Index ", maxIndex)
print("Min Index ", minIndex)
