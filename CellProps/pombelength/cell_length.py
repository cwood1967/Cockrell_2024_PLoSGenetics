# %%

import numpy as np
import scipy.ndimage as ndi
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.linalg import eig

import scipy.ndimage as ndi
from skimage.measure import inertia_tensor, inertia_tensor_eigvals
# %%
def celllength(x, y):
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    
    ds = np.sqrt(dx*dx + dy*dy)
    s = ds.sum()
    return s

def segline(x, m1, b1, m2, p):
    m = np.where(x < p, m1, m2)
    b2 = (m1*p + b1) - (m2*p)
    b =np.where(x < p, b1, b2)
    return m*x + b

def calc_cell_length(points):
    """Calculate the length of the pombe cell by calculating the
    inertia tensor, rotating the major axis to horizontal, and fitting
    the points of the mask to a segmented line
 
    If the length can't be fit to a segmented line, the distance between the extreme
    points along x is returned

    Parameters
    ----------
    points: tuple
        length 2, first element is an array of y points, second
        is an array of x points 
        
    Returns
    -------
    float : the length of the cell
    """
    X = points[1]
    y = points[0]
    ymin = y.min()
    ymax = y.max()  
    xmin = X.min()
    xmax = X.max()

    cmx = X.mean()
    cmy = y.mean()
    X = X - cmx
    y = y - cmy

    try:
        a1, a0 = np.polyfit(X, y, 1)
    except:
        print("Can't polyfit")
        return None

    Ixy = (X*y).sum()
    Ixx = np.sum(y*y)
    Iyy = np.sum(X*X)
    I = np.array([[Iyy, Ixy],
                [Ixy, Ixx]])
    ev = eig(I)
   
    angle = (180/np.pi)*np.arctan2(ev[1][0,1],ev[1][0,0])
    if a1 > 0:
        angle = -angle
    rangle = (90 - angle)
    
    dr = np.pi/180
    dx = xmax - xmin
    if dx < .00001:
        dx = 0.1
    pmx = (ymax - ymin)/(xmax - xmin)

    if np.abs(pmx) > 1.:
        uev = 1
        uev2 = 0
    else:
        uev = 0
        uev2 = 1
         
    Xp = X*ev[1][uev,0] - y*ev[1][uev,1]
    yp = X*ev[1][uev,1] + y*ev[1][uev,0]
    
    if (yp.max() - yp.min()) > (Xp.max() - Xp.min()):
        uev = uev2
        Xp = X*ev[1][uev2,0] - y*ev[1][uev2,1]
        yp = X*ev[1][uev2,1] + y*ev[1][uev2,0]
         
    ux = np.sort(np.unique(Xp))

    s0 = .1 
    b0 = [-1, -2, -1, .75*Xp.min()]
    bf = [1, 2, 1, .75*Xp.max()]

    try:
        p, _ = curve_fit(segline, Xp, yp, p0=[s0, 0.2, -s0, -5 + ux.mean()],
                        bounds=(b0, bf))
        upy = segline(ux, *p)
        pc = segline([ux[0], p[3], ux[-1]], *p)
        ds1 = np.sqrt((ux[0] - p[3])**2 + (pc[0] - pc[1])**2)
        ds2 = np.sqrt((p[3] - ux[-1])**2 + (pc[2] - pc[1])**2)
        length = ds1 + ds2
    except Exception as err:
        print(err)
        print(b0, bf, (s0, -s0, ux.mean()) )
        length = np.abs(ux[0] - ux[-1])
        upy = ux

    return {'length':length, 'xline':ux, 'yline':upy,
            'eigenvec':ev[1][uev], 'moi':I, 'Xp':Xp, 'yp':yp}