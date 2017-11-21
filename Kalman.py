import numpy as np

def predict(X,P,A,Q,B=0,U=0):
    if B == 0:
        B = np.zeros(X.shape[0])
    if U == 0:
        U = np.zeros((X.shape[0],1))
    X = np.dot(A,X) + np.dot(B,U)
    P = np.dot(A,np.dot(P,A.T)) + Q
    return (X,P)

def update(P,H,R,X,Z):
    K = np.dot(P,np.dot(H.T,np.linalg.inv(np.dot(H,np.dot(P,H.T)) + R)))
    X = X + np.dot(K,(Z - np.dot(H,X)))
    P = np.dot((np.eye(P.shape[0]) - np.dot(K,H)),P)
    return (X,P)
