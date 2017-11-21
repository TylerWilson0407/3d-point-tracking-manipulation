def linearTriangulate(P_R, P_L, projPointsR, projPointsL):
    import cv2
    import numpy as np

    points3D = np.zeros((3,projPointsR.shape[1]))

    for i in range(projPointsR.shape[1]):
        rx = projPointsR[0,i]
        ry = projPointsR[1,i]

        lx = projPointsL[0,i]
        ly = projPointsL[1,i]

        A = ((rx*P_R[2,0]-P_R[0,0],    rx*P_R[2,1]-P_R[0,1],      rx*P_R[2,2]-P_R[0,2]),
            (ry*P_R[2,0]-P_R[1,0],    ry*P_R[2,1]-P_R[1,1],      ry*P_R[2,2]-P_R[1,2]),
            (lx*P_L[2,0]-P_L[0,0], lx*P_L[2,1]-P_L[0,1],   lx*P_L[2,2]-P_L[0,2]),
            (ly*P_L[2,0]-P_L[1,0], ly*P_L[2,1]-P_L[1,1],   ly*P_L[2,2]-P_L[1,2]))
        
        A = np.asarray(A)

        B = (-(rx*P_R[2,3]    -P_R[0,3]),
        -(ry*P_R[2,3]  -P_R[1,3]),
        -(lx*P_L[2,3]    -P_L[0,3]),
        -(ly*P_L[2,3]    -P_L[1,3]))

        B = np.asarray(B)

        X = np.zeros((3))

        _,X = cv2.solve(A ,B, X, cv2.DECOMP_SVD)

        points3D[:,i] = X

    return points3D

def polyTriangulate(F, projPointsR, projPointsL):
    import cv2
    import numpy as np

    rx = projPointsR[0,i]
    ry = projPointsR[1,i]

    lx = projPointsL[0,i]
    ly = projPointsL[1,i]

    T_R = ((1, 0, -rx),
           (0, 1, -ry),
           (0, 0, 1))

    T_R = np.asarray(T_R)

    T_L = ((1, 0, -lx),
           (0, 1, -ly),
           (0, 0, 1))

    T_L = np.asarray(T_L)

    F = np.linalg.inv(T_L).T*F*np.linalg.inv(T_R)

    e_R = zeros((3))
    e_L = zeros((3))

    _,e_R = cv2.solve(F ,np.zeros((3)), e_R, cv2.DECOMP_SVD)

    _,e_L = cv2.solve(F ,np.zeros((3)), e_L, cv2.DECOMP_SVD)








    
