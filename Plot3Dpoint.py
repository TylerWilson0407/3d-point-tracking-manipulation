import pyqtgraph
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np

def Initialize():

    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.opts['distance'] = 800
    w.show()
    w.setWindowTitle('Tracked Position')
    w.move(20,600)

    g = gl.GLGridItem()
    g.scale(25, 25, 1)
    w.addItem(g)

    ### finger points
    pos = np.array([[1,0,0],
                    [0,1,0],
                    [0,0,1]])

    size = np.array([10,10,10])
    color = np.array([[1.0, 0.0, 0.0, 0.8],
                      [0.0, 0.0, 1.0, 0.8],
                      [0.0, 1.0, 0.0, 0.8]])

    SP = gl.GLScatterPlotItem(pos=pos, size=size, color=color, pxMode=False)
    w.addItem(SP)

    ### heading vector
    pos = np.array([[0, 0, 0],
                     [1, 1, 1]])
    width = 2
    color = np.array([[0.25, 0.25, 0.75, 1],
                      [0.25, 0.25, 0.75, 1]])

    LPh = gl.GLLinePlotItem(pos=pos, color=color, width=width, antialias=True)
    w.addItem(LPh)

    ### roll vector
    pos = np.array([[0, 0, 0],
                    [1, 1, 1]])
    width = 2
    color = np.array([[0.25, 0.75, 0.25, 1],
                      [0.25, 0.75, 0.25, 1]])

    LPr = gl.GLLinePlotItem(pos=pos, color=color, width=width, antialias=True)
    w.addItem(LPr)

    ### joint positions
    pos = np.zeros((7,3))

    width = 3
    size = 5
    colorline = np.array([[0.5, 0.5, 0.5, 1]])
    colorline = np.tile(colorline,(8,1,1))
    
    color = np.array([[0.25, 0.25, 0.75, 1],
                      [0.75, 0.25, 0.25, 1],
                      [0.25, 0.75, 0.25, 1],
                      [0.25, 0.25, 0.75, 1],
                      [0.75, 0.25, 0.25, 1],
                      [0.25, 0.75, 0.25, 1],
                      [0.25, 0.25, 0.75, 1],
                      [0.75, 0.25, 0.25, 1]])

    armLines = gl.GLLinePlotItem(pos=pos, color=colorline,
                                 width = width, antialias=True)
    jointPos = gl.GLScatterPlotItem(pos=pos, size=size, color=color, pxMode=False)
    w.addItem(armLines)
    w.addItem(jointPos)

    ### joint coordinate systems

    p_0 = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    width = 3

    color = np.array([[0.75, 0.25, 0.25, 1],
                      [0.25, 0.75, 0.25, 1],
                      [0.25, 0.25, 0.75, 1]])

    jointCoords = []
    
    for i in range(7):
        jointXYZ = []
        for j in range(3):
            pos = np.array([p_0[:3,j] + p_0[:3,3],
                            p_0[:3,3]])
            print pos
            jXYZ = gl.GLLinePlotItem(pos=pos, color=np.tile(color[j,:],(2,1)),
                                   width = width, antialias=True)
            w.addItem(jXYZ)
            jointXYZ.append(jXYZ)
        jointCoords.append(jointXYZ)

    return app, w, SP, LPh, LPr, armLines, jointPos, jointCoords

def HandUpdate(SP,LPh,LPr,pos,posh,posr):
    SP.setData(pos=pos)
    LPh.setData(pos=posh)
    LPr.setData(pos=posr)

def ArmUpdate(armLines,jointPos, jointCoords,pos,p):
    pos = np.concatenate(([[0,0,0]],pos), axis=0)
    armLines.setData(pos=pos)
    jointPos.setData(pos=pos)

    for i in range(6):
        for j in range(3):
            pos = np.array([p[i,:3,3], 15*p[i,:3,j] + p[i,:3,3]])
            jointCoords[i][j].setData(pos = pos)

def close(w):
    w.destroy()
