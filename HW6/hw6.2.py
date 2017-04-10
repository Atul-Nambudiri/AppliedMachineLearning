from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy import misc, spatial
from math import log, exp
import random
import matplotlib.pyplot as plt
import numpy as np


def e_step(pis, mus, x):
    diff = spatial.distance.cdist(x, mus, 'sqeuclidean')
    w = np.exp(-.5 * diff)
    w *= pis
    s = np.sum(w, axis = 1)
    s = s[:, np.newaxis]
    w /= s
    return w
    
def m_step(x, w, N):
    mus = np.transpose(x).dot(w)
    mus = mus / w.sum(axis = 0)
    mus = np.transpose(mus)
    # mus = mus + 0.00001
    # for i in xrange(w.shape[1]):
    #     mus[i] = mus[i] / np.sum(mus[i])

    pi = np.sum(w, axis = 0)
    pi = pi/N

    return mus,pi

def revert_shape_image(img, original_shape):
    n_image = np.zeros(original_shape)
    for k in xrange(img.shape[0]):
        for i in xrange(original_shape[0]):
            for j in xrange(original_shape[1]):
                n_image[i][j][k] = img[k][i * original_shape[1] + j]
    return n_image

def reshape_image(img, samples):
    n_image = np.zeros((3, samples))
    for k in xrange(3):
        for i in xrange(img.shape[0]):
            for j in xrange(img.shape[1]):
                n_image[k][img.shape[1] * i + j] = img[i][j][k]
    return n_image

def main():
    n_clusters = 10
    img = misc.imread('dataset/RobertMixed03.jpg')
    original_shape = img.shape
    samples = img.shape[0] * img.shape[1]

    img = np.double(reshape_image(img, samples))
    mean = np.mean(img, axis = 1)
    mean = mean[:, np.newaxis]

    img -= mean
    #Scale to range of 0 - 1
    img /= 255
    
    # cov = np.cov(img)
    # img = cov * img * 

    img = img.T

    kmeans = KMeans(n_clusters=10, random_state=0)
    kmeans.fit(img)

    temp_labels = [random.randint(0, 9) for i in range(samples)]

    cluster_mus = np.zeros((n_clusters, 3))
    cluster_weights = np.zeros((n_clusters))

    for index, cluster in enumerate(kmeans.labels_):
        cluster_mus[cluster] += img[index]
        cluster_weights[cluster] += 1

    for i in range(len(cluster_mus)):
        if cluster_weights[i] != 0:
            cluster_mus[i] /= cluster_weights[i]

    cluster_mus += .00001
    cluster_weights += .00001

    cluster_weights /= np.sum(cluster_weights)

    for i in xrange(n_clusters):
        cluster_mus[i] = cluster_mus[i] / np.sum(cluster_mus[i])

    w_old = np.zeros(1)
    w = np.ones(1)

    count = 0
    while(not np.allclose(w,w_old) and count < 15):
        print("One Iteration - ", count)
        count += 1
        w_old = np.copy(w)
        w = e_step(cluster_weights, cluster_mus, img)
        print(w)
        ps, cluster_weights = m_step(img, w, img.shape[0])

        for index, row in enumerate(w):
            top = np.argsort(row)[-1]
            img[index] = cluster_mus[top]

        print(img)

    img *= 255
    img = img.T
    img += mean
    img = revert_shape_image(img, original_shape)
    img = img.astype(np.uint8)
    plt.imshow(img)
    plt.savefig('testplot.png')

if __name__ == '__main__':
    main()
    

