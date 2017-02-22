def MDS(d):
    A = np.identity(10) - (np.multiply(np.ones((10,10)), np.transpose(np.ones((10,10)))))/10
    W = (np.multiply(np.multiply(A, d), np.transpose(A)))/2
    eigenVal, eigenVec = linalg.eig(W)

    i = eigenVal.argsort()[::-1]   
    lamb = eigenVal[i]
    U = eigenVec[:,i]

    A_n = np.sqrt(A[:2][:2])
    U_n = U[:2]

    V = np.multiply(A_n, np.transpose(U_n))
    print(V)
