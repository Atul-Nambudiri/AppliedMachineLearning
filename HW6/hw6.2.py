from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy import misc, spatial
from math import log, exp
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys

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
    mus = mus + 0.00001
    for i in xrange(w.shape[1]):
        mus[i] = mus[i] / np.sum(mus[i])

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


def run_em(img, n_clusters, samples, cluster_labels):
    cluster_mus = np.zeros((n_clusters, 3))
    cluster_weights = np.zeros((n_clusters))

    for index, cluster in enumerate(cluster_labels):
        cluster_mus[cluster] += img[index]
        cluster_weights[cluster] += 1

    for i in range(len(cluster_mus)):
        if cluster_weights[i] != 0:
            cluster_mus[i] /= cluster_weights[i]

    cluster_mus += .00001
    cluster_weights += .00001

    cluster_weights /= np.sum(cluster_weights)

    w_old = np.zeros(1)
    w = np.ones(1)

    count = 0
    while(not np.allclose(w,w_old) and count < 300):
        print("One Iteration - ", count)
        count += 1
        w_old = np.copy(w)
        w = e_step(cluster_weights, cluster_mus, img)
        ps, cluster_weights = m_step(img, w, img.shape[0])

    return w, cluster_mus

def run_6_21():
    n_clusters_list = [10, 20, 50]
    images = ['RobertMixed03.jpg', 'smallstrelitzia.jpg', 'smallsunset.jpg']
    for image_name in images:
        for n_clusters in n_clusters_list:
            print(image_name + " - " + str(n_clusters))
            img = misc.imread('dataset/' + image_name)
            original_shape = img.shape
            samples = img.shape[0] * img.shape[1]

            img = np.double(reshape_image(img, samples))
            #Scale to range of 0 - 1

            mean = np.mean(img, axis = 1)
            mean = mean[:, np.newaxis]
            img -= mean
            img /= 25.5
            img = img.T

            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(img)

            w, cluster_mus = run_em(img, n_clusters, samples, kmeans.labels_)

            for index, row in enumerate(w):
                top = np.argsort(row)[-1]
                img[index] = cluster_mus[top]

            img = img.T
            img *= 25.5
            img += mean
            img = revert_shape_image(img, original_shape)
            img = img.astype(np.uint8)
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            plt.axis('off')
            plt.imshow(img)
            plt.savefig('output/621-' + image_name + '-' + str(n_clusters) + '.png', bbox_inches=extent, pad_inches=0)

def run_6_22():
    total_labels = []
    for count in range(10):
        n_clusters = 20
        image_name = 'dataset/smallsunset.jpg'
        img = misc.imread(image_name)
        original_shape = img.shape
        samples = img.shape[0] * img.shape[1]

        img = np.double(reshape_image(img, samples))
        #Scale to range of 0 - 1

        mean = np.mean(img, axis = 1)
        mean = mean[:, np.newaxis]
        img -= mean
        img /= 25.5
        img = img.T

        kmeans = KMeans(n_clusters=n_clusters, random_state=count + 1)
        kmeans.fit(img)

        total_labels.append(kmeans.labels_)

        w, cluster_mus = run_em(img, n_clusters, samples, kmeans.labels_)

        for index, row in enumerate(w):
            top = np.argsort(row)[-1]
            img[index] = cluster_mus[top]

        img = img.T
        img *= 25.5
        img += mean
        img = revert_shape_image(img, original_shape)
        img = img.astype(np.uint8)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.axis('off')
        plt.imshow(img)
        plt.savefig('output/622-' + str(count) + '.png', bbox_inches=extent, pad_inches=0)
        count += 1

    total_labels = np.array(total_labels)
    print(np.vstack({tuple(row) for row in total_labels}))



def main():
    run_6_21()
    run_6_22()

if __name__ == '__main__':
    main()
