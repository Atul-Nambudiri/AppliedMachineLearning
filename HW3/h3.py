import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from PIL import Image

from collections import defaultdict

def unpickle(file):
    """
    This is from the Cifar website
    """
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def run_pca(data):
    """
    This function runs PCA on the passed in data, and returns the 
    principal components after it is inverse transformed back to its original position
    """
    pca = PCA(n_components=20)
    pca.fit(data)
    return pca.inverse_transform(pca.fit_transform(data))

def run_part_c_pca(data, mean):
    """
    This function runs PCA on the passed in data, and returns the 
    principal components after it is inverse transformed back to its original position.
    It uses the passed in mean to do the inverse transform, as opposed to the mean of the original dataset
    This is usefull for Part c
    """
    pca = PCA(n_components=20)
    res = pca.fit_transform(data)
    
    #Set mean
    pca.mean_ = mean
    return pca.inverse_transform(res)

def calculate_error(A, B):
    """
    Calculates the square error between two data sets
    """
    res = np.subtract(A, B)
    res = np.square(res)
    sum = np.sum(res, 1)
    return np.mean(sum)

def convert_array_to_image(array):
    """
    Converts the image from an array into an image
    """
    image = np.zeros((32, 32, 3))
    for i in range(32):
        for j in range(32):
            for k in range(3):
                image[i, j, k] = array[i * 32 + j + k*1024]/255.0
    return image

def main():
    data = defaultdict(list)
    means = []
    pcas = []
    errors = []
    for i in range(1,6):
        file = "cifar-10-batches-py/data_batch_%d" % i
        res = unpickle(file)
        for j in range(len(res['labels'])):
            data[res['labels'][j]].append(res['data'][j])
    for i in range(10):
        numpy_data = np.asarray(data[i])
        pcas.append(run_pca(numpy_data))
        means.append(np.mean(numpy_data, 0))
    
    # Display mean images
    for i in range(10):
        plt.imshow(convert_array_to_image(means[i]))
        plt.show()
        
    # Find error for each category
    for i in range(10):
        error = calculate_error(np.asarray(data[i]), pcas[i])
        errors.append(error)

    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    #This is to graph the errors for part a
    y = np.arange(1, len(labels) + 1)
    plt.bar(y, errors, align='center', alpha=0.5)
    plt.xticks(y, labels)
    plt.show()
    
    
    #Part b
    D = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            D[i,j] = np.linalg.norm(means[i]-means[j])
    coords = MDS().fit_transform(D)
    
    #from stack overflow to create labels. This is used for the graph for part b
    fig = plt.figure()
    plt.scatter(coords[:, 0], coords[:, 1])
    for label, x, y in zip(labels, coords[:, 0], coords[:, 1]):
        plt.annotate(label, (x, y))
    plt.show()
    
    
    #Part c
    E = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            AB = run_part_c_pca(np.asarray(data[j]), means[i])
            BA = run_part_c_pca(np.asarray(data[i]), means[j])
            EAB = calculate_error(np.asarray(data[i]), AB)
            EBA = calculate_error(np.asarray(data[j]), BA)
            E[i,j] = (.5) * ((EAB) + (EBA))
    coords = MDS().fit_transform(E)
    
    #from stack overflow to create labels. This is used for the graph for part b
    fig = plt.figure()
    plt.scatter(coords[:, 0], coords[:, 1])
    for label, x, y in zip(labels, coords[:, 0], coords[:, 1]):
        plt.annotate(label, (x, y))
    plt.show()

           
if __name__ == '__main__':
    main()