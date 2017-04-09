from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from math import log, exp
import random
import numpy as np

D = 0
W = 0
NNZ = 0

def e_step(pis, ps, x):
    print(ps.shape)
    print(pis.shape)
    print(x.shape)
    w = np.zeros((D, 30))
    for i in xrange(D):
        for j in xrange(30):
            w[i][j] = np.sum(x[i] * np.log(ps[j])) + np.log(pis[j])
            # top = 0
            # for k in xrange(len(ps[0])):
            #     top += log(ps[j][k])*x[i][k]
            # top += log(pis[j])
            # w[i][j] = top
        m = np.amax(w[i])
        w[i] = np.exp(w[i] - m)
        s = np.sum(w[i])
        w[i] = w[i] / s
    return w

    # w = np.zeros((D, 30))
    # for i in xrange(D):
    #     for j in xrange(30):
    #         top = 1
    #         for k in xrange(len(ps[0])):
    #             top *= (ps[j][k])**x[i][k]
    #         top *= pis[j]
    #         w[i][j] = top
    #     s = np.sum(w[i])
    #     for j in xrange(30):
    #         w[i][j] /= s
    #     print("I: %d" % (i))
    # return w
        

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
    p = p / p.max(axis = 0)
    p = np.transpose(p)

    pi = np.sum(w, axis = 0)
    pi = pi/N  
    # sum1 = x[0] * w[0][j]
    # sum2 = np.sum(x[0])*w[0][j]
    # pi = w[0][j]
    # for i in xrange (1, x):
    #     sum1 += x[i] * w[i][j]
    #     sum2 += np.sum(x[i])*w[i][j]
    #     pi += w[i][j]
    # p = sum1/sum2
    # pi = pi/N
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
    numpy_doc_vectors = numpy_doc_vectors + 1
    # kmeans = KMeans(n_clusters=30, random_state=0)
    # kmeans.fit(numpy_doc_vectors)

    temp_labels = [random.randint(0, 29) for i in range(D)]
    # kmeans.labels_
    # When calculating the pis, we dont want to have an values that are 0. 
    # So add a small value to each pi
    pis = [0.000001 for i in xrange(30)]

    for i in temp_labels:
        pis[i] += 1.0

    pis = [i/D for i in pis]

    #Normalize Pi to make sure they add up to 1
    pis = [i/sum(pis) for i in pis]

    pis = np.array(pis)

    ps = np.zeros((30,W))

    for i in xrange(30):
        ps[i] += 0.000001

    for index, cluster in enumerate(temp_labels):
        ps[cluster] = np.add(ps[cluster], numpy_doc_vectors[index])

    for i in xrange(30):
        ps[i] = ps[i] / np.sum(ps[i])

    w_old = np.zeros(1)
    w = np.ones(1)

    count = 0
    while(not np.array_equal(w,w_old)):
        print("One Iteration - ", count)
        count += 1
        w_old = np.copy(w)
        w = e_step(pis, ps, numpy_doc_vectors)
        print(w)
        # Q = Q_step(numpy_doc_vectors, ps, pis, w)
        ps, pis = m_step(numpy_doc_vectors, w, D, 1)

if __name__ == '__main__':
    main()
    

