def InitializeKinematics():
    import numpy as np
    
    ## define DH parameters/geometry
    ## lengths in mm
    L1 = 0
    L2 = 90.3
    L3 = 87.81
    L4 = 8
    L5 = 0
    L6 = 0


    d1 = 81.9
    d2 = 0
    d3 = 0
    d4 = 0
    d5 = 77
    d6 = 78.5

    alpha = np.asarray([-90, 0, 0, 90, -90, 0], dtype = np.float)
    a = np.asarray([L1, L2, L3, L4, L5, L6], dtype = np.float)
    d = np.asarray([d1, d2, d3, d4, d5, d6], dtype = np.float)
    alpha *= np.pi/180

def IKsolve(theta,Xg):
    import numpy as np

    e = np.array([1, 1, 1, 0.05, 0.05, 0.05])

    a_max = np.array([60, 90, 225, 180, 90, 90])
    a_min = np.array([-60, 0, 45, 0, -90, -90])

    theta *= np.pi/180

    Z = np.empty((6,4,4))
    X = np.empty((6,4,4))
    T = np.empty((6,4,4))
    p = np.empty((6,4,4))
    T_n = np.eye(4)

    for i in range(6):
        Z[i] = [[np.cos(theta[i]), -np.sin(theta[i]), 0, 0],
                [np.sin(theta[i]), np.cos(theta[i]), 0, 0],
                [0, 0, 1, d[i]],
                [0, 0, 0, 1]]
        X[i] = [[1, 0, 0, a[i]],
                [0, np.cos(alpha[i]), -np.sin(alpha[i]), 0],
                [0, np.sin(alpha[i]), np.cos(alpha[i]), 0],
                [0, 0, 0, 1]]
        T[i] = np.dot(Z[i], X[i])
        T_n = np.dot(T_n, T[i])
        p[i] = T_n

    ## current position

    X = np.empty((6))

    X[:3] = p[5,:3,3]
    X[3:] = p[5,:3,2]

    dX = Xg - X

    ## Jacobian

    J = np.empty((6,6))

    ##for i in range(6):
    for j in range(6):
        vj = p[j,:3,2]
        pj = p[j,:3,3]
        si = p[5,:3,3]
        J[:3,j] = np.cross(vj,(si - pj))
        J[3:,j] = vj

    Jt = np.transpose(J)

    Ji = np.dot(Jt,np.linalg.inv(np.dot(J,Jt)))

    err = np.abs(np.dot((np.eye(6) - np.dot(J,Ji)), dX))

    while (err > e).any():
        dX = dX / 2
        err = np.abs(np.dot((np.eye(6) - np.dot(J,Ji)), dX))

    theta = theta + np.dot(Ji, dX)
    theta *= 180/np.pi

    theta = np.maximum(a_min, theta)
    theta = np.minimum(a_max, theta)

    arm_pos = p[:,:3,3]

    return theta, arm_pos





