import numpy as np
import bempp.api
import scipy.optimize as SciOp

def line(J,alpha=1/3):
    X = [np.array([0,1])]
    Y =  []
    if J > 0:
        for j in range(J):
            Y =  []
            for (a,b) in X:
                width = b-a
                x2 = a + width*alpha
                x3 = b - width*alpha
                Y.append(np.array([a, x2]))
                Y.append(np.array([x3, b]))
            X = Y
    else:
        Y = X
    return Y

def dust(J,h_max=np.Inf,M=0,middle=1/3):
    X = line(J, middle)
    Y = line(J, middle)
    # now compute the meshwidth h
    if h_max<np.Inf:
        h=h_max
    else:
        h=((1-middle)/2)**(J+M)


    S = []
    for x in X:
        for y in Y:
            v = np.array([[x[0],y[0], 0.0],
                [x[0],y[1], 0.0],
                [x[1],y[1], 0.0],
                [x[1],y[0], 0.0]])
            s = bempp.api.shapes.screen(v,h)
            S.append(s)
            # S.append([(x[0],y[0]),(x[0],y[1]),(x[1],y[1]),(x[1],y[0])])
    return bempp.api.grid.union(S)
    
