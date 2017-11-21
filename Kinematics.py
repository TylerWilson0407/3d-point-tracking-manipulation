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

def InitializeKinematicChain():

    alpha = np.asarray([-90, 0, 0, 90, -90, 0], dtype = np.float) # alpha array
    a = np.asarray([0, 90.3, 87.8, 8, 0, 0], dtype = np.float) # a array (common normal length)
    d = np.asarray([81.9, 0, 0, 0, 77, 78.5], dtype = np.float) # d array (z offset length)
    epsilon = np.array([1, 1, 1, 0.05, 0.05, 0.05]) # max error of pseudo inverse
    a_min = np.array([-60, -90, 45, 0, -90, -90]) # minimum joint angle boundaries
    a_max = np.array([60, 0, 225, 180, 90, 90]) # maximum joint angle boundaries
    D_max = 5 # maximum change in position per iteration (in mm)

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

    arm_pos = p[:,:3,3]

    return arm_pos

def IKsolve(KC,theta,Xg):

    theta *= np.pi/180

    T = np.empty((6,4,4))
    p = np.empty((6,4,4))
    T_n = np.eye(4)

    for i in range(6):
        T[i] = [[np.cos(theta[i]), -np.sin(theta[i])*np.cos(KC.alpha[i]),
                 np.sin(theta[i])*np.sin(KC.alpha[i]), KC.a[i]*np.cos(theta[i])],
                [np.sin(theta[i]), np.cos(theta[i])*np.cos(KC.alpha[i]),
                 -np.cos(theta[i])*np.sin(KC.alpha[i]), KC.a[i]*np.sin(theta[i])],
                [0, np.sin(KC.alpha[i]), np.cos(KC.alpha[i]), KC.d[i]],
                [0, 0, 0, 1]]
        T_n = np.dot(T_n,T[i])
        p[i] = T_n

    ## current position

    X = np.empty((6))

    X[:3] = p[5,:3,3]
    X[3:] = p[5,:3,2]

    dX = Xg - X

    ## Jacobian

    J = np.empty((6,6))
    
    for j in range(6):
        vj = p[j,:3,2]
        pj = p[j,:3,3]
        si = p[5,:3,3]
        J[:3,j] = np.cross(vj,(si - pj))
        J[3:,j] = vj

    Jt = np.transpose(J)

    Ji = np.dot(Jt,np.linalg.inv(np.dot(J,Jt)))

    err = np.abs(np.dot((np.eye(6) - np.dot(J,Ji)), dX))

    while (err > KC.epsilon).any():
        dX = dX / 2
        err = np.abs(np.dot((np.eye(6) - np.dot(J,Ji)), dX))

    theta = theta + np.dot(Ji, dX)
    theta *= 180/np.pi

    theta = np.maximum(KC.a_min, theta)
    theta = np.minimum(KC.a_max, theta)

    arm_pos = p[:,:3,3]

    return theta, arm_pos

def IKsolve3(KC,theta,Xg):

    theta *= np.pi/180

    T = np.empty((6,4,4))
    p = np.empty((6,4,4))
    T_n = np.eye(4)

    for i in range(6):
        T[i] = [[np.cos(theta[i]), -np.sin(theta[i])*np.cos(KC.alpha[i]),
                 np.sin(theta[i])*np.sin(KC.alpha[i]), KC.a[i]*np.cos(theta[i])],
                [np.sin(theta[i]), np.cos(theta[i])*np.cos(KC.alpha[i]),
                 -np.cos(theta[i])*np.sin(KC.alpha[i]), KC.a[i]*np.sin(theta[i])],
                [0, np.sin(KC.alpha[i]), np.cos(KC.alpha[i]), KC.d[i]],
                [0, 0, 0, 1]]
        T_n = np.dot(T_n,T[i])
        p[i] = T_n

    ## current position

    X = p[5,:3,3]

    dX = Xg[:3] - X
    dX_norm = np.linalg.norm(dX)

    if dX_norm > KC.D_max:
        dX *= KC.D_max / dX_norm

    ## Jacobian

    J = np.empty((3,6))
    
    for j in range(6):
        vj = p[j,:3,2]
        pj = p[j,:3,3]
        si = p[5,:3,3]
        J[:,j] = np.cross(vj,(si - pj))

##    Jt = np.transpose(J)

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

    return theta, arm_pos

def IKsolveDLS3(KC,theta,Xg):

    theta *= np.pi/180

    T = np.empty((6,4,4))
    p = np.empty((6,4,4))
    T_n = np.eye(4)

    for i in range(6):
        T[i] = [[np.cos(theta[i]), -np.sin(theta[i])*np.cos(KC.alpha[i]),
                 np.sin(theta[i])*np.sin(KC.alpha[i]), KC.a[i]*np.cos(theta[i])],
                [np.sin(theta[i]), np.cos(theta[i])*np.cos(KC.alpha[i]),
                 -np.cos(theta[i])*np.sin(KC.alpha[i]), KC.a[i]*np.sin(theta[i])],
                [0, np.sin(KC.alpha[i]), np.cos(KC.alpha[i]), KC.d[i]],
                [0, 0, 0, 1]]
        T_n = np.dot(T_n,T[i])
        p[i] = T_n

    ## current position

    X = p[5,:3,3]

    dX = Xg[:3] - X
    dX_norm = np.linalg.norm(dX)

    if dX_norm > KC.D_max:
        dX *= KC.D_max / dX_norm

    ## Jacobian

    J = np.empty((3,6))
    
    for j in range(6):
        vj = p[j,:3,2]
        pj = p[j,:3,3]
        si = p[5,:3,3]
        J[:,j] = np.cross(vj,(si - pj))

    np.set_printoptions(suppress=True,precision=2)
    print p
    print J

    lam = 0.01

    d_theta = np.dot(np.dot(J.T,np.linalg.inv(np.dot(J,J.T) + lam**2*np.eye(3))), dX)

    theta = theta + d_theta
    theta *= 180/np.pi

##    theta = np.maximum(KC.a_min, theta)
##    theta = np.minimum(KC.a_max, theta)

    arm_pos = p[:,:3,3]

    return theta, arm_pos
