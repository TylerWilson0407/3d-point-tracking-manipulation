import numpy as np

class KinematicChain:
    def __init__(self, alpha, a, d, epsilon, a_min, a_max, D_max):
        self.alpha = alpha * np.pi/180
        self.a = a
        self.d = d
        self.epsilon = epsilon
        self.a_min = a_min
        self.a_max = a_max
        self.D_max = D_max

##def InitializeKinematicChain():
##
##    alpha = np.asarray([0, -90, 0, 0, 90, -90, 0], dtype = np.float) # alpha array
##    a = np.asarray([0, 0, 90.3, 87.8, 8, 0, 0], dtype = np.float) # a array (common normal length)
##    d = np.asarray([0, 81.9, 0, 0, 0, 77, 78.5], dtype = np.float) # d array (z offset length)
##    epsilon = np.array([10e-6, 10e-6, 10e-6, 10e-6, 10e-6, 10e-6], dtype = np.float) # max error of pseudo inverse
##    a_min = np.array([0, -90, -180, -45, 0, -180, -90], dtype = np.float) # minimum joint angle boundaries
##    a_max = np.array([0, 90, 0, 135, 180, 0, 90], dtype = np.float) # maximum joint angle boundaries
##    D_max = np.array([5, 0.2], dtype = np.float) # maximum change in position per iteration (in mm)
##
##    KC = KinematicChain(alpha, a, d, epsilon, a_min, a_max, D_max)
##
##    return KC

def InitializeKinematicChain2():

    alpha = np.asarray([90, 0, -90, 90, -90, 0], dtype = np.float) # alpha array
    a = np.asarray([0, 50, 50, 0, 0, 0], dtype = np.float) # a array (common normal length)
    d = np.asarray([67, 0, 15, 0, 0, 0], dtype = np.float) # d array (z offset length)
    epsilon = np.array([10e-6, 10e-6, 10e-6, 10e-6, 10e-6, 10e-6], dtype = np.float) # max error of pseudo inverse
    a_min = np.array([-90, -180, -45, 0, -180, -90], dtype = np.float) # minimum joint angle boundaries
    a_max = np.array([90, 0, 135, 180, 0, 90], dtype = np.float) # maximum joint angle boundaries
    D_max = np.array([5, 0.2], dtype = np.float) # maximum change in position per iteration (in mm)

    KC = KinematicChain(alpha, a, d, epsilon, a_min, a_max, D_max)

    return KC


def FKsolve(KC,theta):

    alpha = KC.alpha
    a = KC.a
    d = KC.d

    theta *= np.pi/180

    T = np.empty((6,4,4))
    p = np.empty((6,4,4))
    T_n = np.eye(4)

    for i in range(6):
        T[i] = [[np.cos(theta[i]), -np.sin(theta[i])*np.cos(alpha[i]),
                 np.sin(theta[i])*np.sin(alpha[i]), a[i]*np.cos(theta[i])],
                [np.sin(theta[i]), np.cos(theta[i])*np.cos(alpha[i]),
                 -np.cos(theta[i])*np.sin(alpha[i]), a[i]*np.sin(theta[i])],
                [0, np.sin(alpha[i]), np.cos(alpha[i]), d[i]],
                [0, 0, 0, 1]]
        T_n = np.dot(T_n,T[i])
        p[i] = T_n

    return p

def IKsolve(KC,theta,Xg):

    p = FKsolve(KC, theta)

    ## current position

    X = np.empty((6))

    X[:3] = p[5,:3,3]
    X[3:] = p[5,:3,2]

    dX = Xg - X
    
    dX_norm_pos = np.linalg.norm(dX[:3])
    dX_norm_orient = np.linalg.norm(dX[3:])

    if dX_norm_pos > KC.D_max[0]:
        dX[:3] *= KC.D_max[0] / dX_norm_pos
    if dX_norm_orient > KC.D_max[1]:
        dX[3:] *= KC.D_max[1] / dX_norm_orient

    ## Jacobian

    J = np.empty((6,6))
    
    for j in range(6):
        vj = p[j,:3,2]
        pj = p[j,:3,3]
        si = p[5,:3,3]
        J[:3,j] = np.cross(vj,(si - pj))
        J[3:,j] = vj

    Ji = np.dot(J.T,np.linalg.inv(np.dot(J,J.T)))

    err = np.abs(np.dot((np.eye(6) - np.dot(J,Ji)), dX))

    while (err > KC.epsilon).any():
        dX = dX / 2
        err = np.abs(np.dot((np.eye(6) - np.dot(J,Ji)), dX))

    theta = theta + np.dot(Ji, dX)
    theta *= 180/np.pi

    theta = np.maximum(KC.a_min, theta)
    theta = np.minimum(KC.a_max, theta)

    arm_pos = p[:,:3,3]

    return theta, arm_pos, p

def IKsolve3(KC,theta,Xg):

    p = FKsolve(KC, theta)

    dX = Xg[:3] - p[5,:3,3]
    dX_norm = np.linalg.norm(dX)

    if dX_norm > KC.D_max[0]:
        dX *= KC.D_max[0] / dX_norm

    ## Jacobian

    J = np.empty((3,6))
    
    for j in range(6):
        vj = p[j,:3,2]
        pj = p[j,:3,3]
        si = p[5,:3,3]
        J[:,j] = np.cross(vj,(si - pj))

    Ji = np.dot(J.T,np.linalg.inv(np.dot(J,J.T)))

    err = np.abs(np.dot((np.eye(3) - np.dot(J,Ji)), dX))

    while (err > KC.epsilon[:3]).any():
        dX /= 2
        err = np.abs(np.dot((np.eye(3) - np.dot(J,Ji)), dX))

    theta = theta + np.dot(Ji, dX)
    theta *= 180/np.pi

##    theta = np.maximum(KC.a_min, theta)
##    theta = np.minimum(KC.a_max, theta)

    arm_pos = p[:,:3,3]

    return theta, arm_pos, p

def IKsolveDLS(KC,theta,Xg):

    p = FKsolve(KC, theta)

    ## current position

    X = np.empty((6))
    X[:3] = p[5,:3,3]
    X[3:] = p[5,:3,2]

    dX = Xg - X
    
    dX_norm_pos = np.linalg.norm(dX[:3])
    dX_norm_orient = np.linalg.norm(dX[3:])

    if dX_norm_pos > KC.D_max[0]:
        dX[:3] *= KC.D_max[0] / dX_norm_pos
    if dX_norm_orient > KC.D_max[1]:
        dX[3:] *= KC.D_max[1] / dX_norm_orient

    ## Jacobian

    J = np.empty((6,6))
    
    for j in range(6):
        vj = p[j,:3,2]
        pj = p[j,:3,3]
        si = p[5,:3,3]
        J[:3,j] = np.cross(vj,(si - pj))
        J[3:,j] = vj
    
    lam = 1000 / (dX_norm_pos*dX_norm_orient**2)

    d_theta = np.dot(np.dot(J.T,np.linalg.inv(np.dot(J,J.T) + lam**2*np.eye(6))), dX)

    theta = theta + d_theta
    theta *= 180/np.pi

    np.set_printoptions(suppress=True,precision=4)
    print p

##    theta = np.maximum(KC.a_min, theta)
##    theta = np.minimum(KC.a_max, theta)

    arm_pos = p[:,:3,3]

    return theta, arm_pos, p

def IKsolveDLS3(KC,theta,Xg):

    p = FKsolve(KC, theta)

    ## current position

    X = p[5,:3,3]

    dX = Xg[:3] - X

    dX = dX / 2
    
    dX_norm_pos = np.linalg.norm(dX[:3])
    dX_norm_orient = np.linalg.norm(dX[3:])

    if dX_norm_pos > KC.D_max[0]:
        dX[:3] *= KC.D_max[0] / dX_norm_pos
    if dX_norm_orient > KC.D_max[1]:
        dX[3:] *= KC.D_max[1] / dX_norm_orient

    ## Jacobian

    J = np.empty((3,6))
    
    for j in range(6):
        vj = p[j,:3,2]
        pj = p[j,:3,3]
        si = p[5,:3,3]
        J[:,j] = np.cross(vj,(si - pj))

    lam = np.maximum(2, 200/dX_norm_pos**2)

    print dX

    d_theta = np.dot(np.dot(J.T,np.linalg.inv(np.dot(J,J.T) + lam**2*np.eye(3))), dX)

    theta = theta + d_theta
    theta *= 180/np.pi

##    theta = np.maximum(KC.a_min, theta)
##    theta = np.minimum(KC.a_max, theta)

    arm_pos = p[:,:3,3]

    return theta, arm_pos, p

def IKsolveSVD(KC,theta,Xg):

    p = FKsolve(KC, theta)

    ## current position

    X = np.empty((6))
    X[:3] = p[5,:3,3]
    X[3:] = p[5,:3,2]

    dX = Xg - X

    dX = dX / 2
    
    dX_norm_pos = np.linalg.norm(dX[:3])
    dX_norm_orient = np.linalg.norm(dX[3:])

    if dX_norm_pos > KC.D_max[0]:
        dX[:3] *= KC.D_max[0] / dX_norm_pos
    if dX_norm_orient > KC.D_max[1]:
        dX[3:] *= KC.D_max[1] / dX_norm_orient

    ## Jacobian

    J = np.empty((6,6))
    
    for j in range(6):
        vj = p[j,:3,2]
        pj = p[j,:3,3]
        si = p[5,:3,3]
        J[:3,j] = np.cross(vj,(si - pj))
        J[3:,j] = vj
    
    lam = 500 / (dX_norm_pos*dX_norm_orient)

    d_theta = np.dot(np.dot(J.T,np.linalg.inv(np.dot(J,J.T) + lam**2*np.eye(6))), dX)

    theta = theta + d_theta
    theta *= 180/np.pi

    np.set_printoptions(suppress=True,precision=4)
    print dX_norm_pos
    print dX_norm_orient
    print lam

    theta = np.maximum(KC.a_min, theta)
    theta = np.minimum(KC.a_max, theta)

    arm_pos = p[:,:3,3]

    return theta, arm_pos, p
