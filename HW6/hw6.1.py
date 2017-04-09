from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from math import log, exp
import random
import matplotlib.pyplot as plt
import numpy as np

D = 0
W = 0
NNZ = 0

def e_step(pis, ps, x):
    w = x.dot(np.transpose(np.log(ps)))  + np.log(pis)
    m = np.max(w, axis = 1)
    m = m[:, np.newaxis]
    w = np.exp(w - m)
    s = np.sum(w, axis = 1)
    s = s[:, np.newaxis]
    w = w / s   
    return w     

def Q_step(x, ps, pis, w):
    Q = 0
    for i in xrange(w):
        for j in xrange(w[0]):
            sum1 = 0
            for k in xrange (x[i]):
                sum1 += x[i][k] * log(ps[j][k])
            sum1 += log(pis[j])
            Q += sum1 * w[i][j]
    return Q
    
def m_step(x, w, N, j):
    p = np.transpose(x).dot(w)
    p = p / p.sum(axis = 0)
    p = np.transpose(p)
    p = p + 0.000001
    for i in xrange(30):
        p[i] = p[i] / np.sum(p[i])

    pi = np.sum(w, axis = 0)
    pi = pi/N  
    pi = pi + 0.000001
    pi /= np.sum(pi)

    return p,pi

def main():
    global D, W, NNZ
    with open('dataset/vocab.nips.txt', 'r') as vocab:
        vocablines = vocab.read().splitlines()

    with open('dataset/docword.nips.txt', 'r') as docword:
        temp_docwordlines = docword.read().splitlines()
        D = int(temp_docwordlines[0])
        W = int(temp_docwordlines[1])
        NNZ = int(temp_docwordlines[2])
        docwordlines = temp_docwordlines[3:]
        
    for i in xrange(len(docwordlines)):
        docwordlines[i]    = docwordlines[i].strip(' ').split(' ')
        docwordlines[i][0] = int(docwordlines[i][0])
        docwordlines[i][1] = int(docwordlines[i][1])
        docwordlines[i][2] = int(docwordlines[i][2])

    #Create the document word count vectors
    document_vectors = [[0 for i in xrange(W)] for i in xrange(D)]

    for docword in docwordlines:
        document_vectors[docword[0] - 1][docword[1] - 1] = docword[2]

    #Run K means to get initial clusters/cluster centers
    numpy_doc_vectors = np.array(document_vectors)
    kmeans = KMeans(n_clusters=30, random_state=0)
    kmeans.fit(numpy_doc_vectors)

    temp_labels = [random.randint(0, 29) for i in range(D)]
    # kmeans.labels_
    # When calculating the pis, we dont want to have an values that are 0. 
    # So add a small value to each pi
    pis = [0.000001 for i in xrange(30)]

    for i in kmeans.labels_:
        pis[i] += 1.0

    pis = [i/D for i in pis]

    #Normalize Pi to make sure they add up to 1
    pis = [i/sum(pis) for i in pis]
    pis = np.array(pis)

    ps = np.zeros((30,W))

    ps = ps + 0.000001

    for index, cluster in enumerate(kmeans.labels_):
        ps[cluster] = np.add(ps[cluster], numpy_doc_vectors[index])

    for i in xrange(30):
        ps[i] = ps[i] / np.sum(ps[i])

    w_old = np.zeros(1)
    w = np.ones(1)

    count = 0
    while(not np.allclose(w,w_old)):
        print("One Iteration - ", count)
        count += 1
        w_old = np.copy(w)
        w = e_step(pis, ps, numpy_doc_vectors)
        # Q = Q_step(numpy_doc_vectors, ps, pis, w)
        ps, pis = m_step(numpy_doc_vectors, w, D, 1)
        for row in ps:
            top_10 = np.argsort(row)[::-1][:10]
            print(",".join([vocablines[i] for i in top_10]))

    plt.plot(np.arange(1, len(pis) + 1), pis)
    plt.xlabel('Topics')
    plt.ylabel('Probability')
    plt.title('Probabilities with which topics are selected')
    plt.show()
    plt.figure()

    for row in ps:
        top_10 = np.argsort(row)[::-1][:10]
        print(",".join([vocablines[i] for i in top_10]))
    print(np.sum(ps))
    print(np.sum(pis))
    print(np.sum(w))

if __name__ == '__main__':
    main()
    

