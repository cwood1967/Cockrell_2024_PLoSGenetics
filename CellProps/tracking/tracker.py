from re import I
import time

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, cdist


class Tracker:
    """ A class to do simple tracking using the center of mass of objects
    
    Parameters
    ----------

    df : Dataframe
        a pandas dataframe with columns for x, y, and time (frame)
        
    frame_col : str
        the name of the column for frames. Defaults to 'frame'.
    
    dmax : float
        The maximum separation for objects in consecutive frames to be
        considered the same object. Defaults to 10.
        
    xm : str
        The column for the x coordinate of each object. Defaults to 'XM'.
        
    ym : str
        The column for the y coordinate of each object. Defaults to 'YM'.

    max_lost : int
        If an object is lost between frame, how many frame to keep looking
        for it. Defaults to 5.
        
    start_frame : int
        What frame to start tracking on. Defaults to 1.
    
   
    Attributes
    ----------
    linkdf : DataFrame 
        Dataframe used for link objects between frames

    groups : GroupBy
        pandas groupby object grouped by frame
        
    """

    def __init__(self, df, frame_col='frame', max_separation=10,
                 xm='XM', ym='YM', max_lost=5, start_frame=1):

        self.df = df
        self.groups = df.groupby(frame_col)
        self.frame_col = frame_col
        self.dmax = max_separation
        self.xm = xm
        self.ym = ym
        self.max_lost = max_lost
        self.start_frame = start_frame

        
    def linkframes(self, frame1, frame2):
        """Links objects between frames by calculating distances between
        objects and linking objects by nearest distance.

        Parameters
        ----------
        frame1 : int 
            The index of the frame
        frame2 : int
            The index of the next frame

        Returns
        -------
        resdf : DataFrame
            A DataFrame with columns for indices of linked object
        ilist : list
            List of objects lost in between frame1 and frame2
        """

        f1 = frame1
        f2 = frame2
        
        df = self.df
        d1 = df[((df.loc[:,self.frame_col] == f1) & (df.loc[:, 'lost'] == 0))
                | (df.loc[:, 'lost'] > 0)]
        d2 = df[df.loc[:,self.frame_col] == f2]
        
        xa = np.array([d1[self.xm], d1[self.ym]]).T
        xb = np.array([d2[self.xm], d2[self.ym]]).T

        if len(d2) == 0:
            return None, None
        dmx = cdist(xa, xb)

        amins = dmx.argmin(axis=1)
        rmins = dmx.min(axis=1)
        w1= np.where(rmins < self.dmax)[0]
        wlost = np.where(rmins >= self.dmax)[0]
        w2 = amins[w1]
        if len(d2) < 600:
            uw2, ucounts = np.unique(w2, return_counts=True)
        
        i1 = d1.iloc[w1].reset_index()
        i2 = d2.iloc[w2].reset_index()
        ilist = list(d1.iloc[wlost].index)
        resdf = pd.DataFrame({'index1':i1['index'], 'index2':i2['index']})
        return resdf, ilist

        
    def run(self):
        """Run the tracking on the DataFrame self.df. This method will add
        the following columns to the original DataFrame:
        
        New Columns
        -----------
        cellid : int
            the id of the tracked object
        keyed : Boolean
            True if the row in the dataframe is in a tracked cell
        nslices : int
            how many frames are in the tracked object
        startframe : int
            the starting frame of the object
        """

        linkdict = dict()
        _lost = []
        num_lost = 0
        self.df['lost'] = 0
        t1 = time.time()
        for j in range(self.df[self.frame_col].min(), self.df[self.frame_col].max() - 1):
            i = j + self.start_frame
            _df,_lost = self.linkframes(i, i + 1)
            if _lost is not None:
                self.df.loc[self.df.index.isin(_lost), 'lost'] += 1
                num_lost += len(_lost)
            if _df is not None:
                linkdict[j] = _df
                self.df.loc[self.df.index.isin(_df.index1), 'lost'] = -1
            self.df.loc[self.df.lost > self.max_lost, 'lost'] = -2
        t2 = time.time()
        # print(f"Link loop: {t2 - t1}, num_lost:{num_lost}")

        t1 = time.time()
        self.linkdf = pd.concat(linkdict.values())
        t2 = time.time()
        # print(f"Concat: {t2 - t1}")

        t1 = time.time()
        self.make_celldict()
        t2 = time.time()
        # print(f"Celldict: {t2 - t1}")

        self.df['keyed'] = False
        counter = 0
        self.df['cell_frame'] = 0 
        t1 = time.time()
        for k, v in self.celldict.items():
            #if counter % 1000  == 0:
            #    print(counter)
            self.df.loc[v, 'cellid'] = k
            self.df.loc[v, 'keyed'] = True
            self.df.loc[v, 'nslices'] = len(v)
            self.df.loc[v, 'startframe'] = self.df.loc[np.min(v), 'frame']
            
            _df = self.df.loc[v].sort_values(self.frame_col)
            counter += 1

        t2 = time.time()
        # print(f"Dataframe: {t2 - t1}")


    def make_celldict(self):        
        """Make a dictionary of cells from the linked frames.
        """

        x1 = self.linkdf.index1.to_numpy()
        x2 = self.linkdf.index2.to_numpy()    
        celldict = dict()
        idinc = 0

        while len(x1) > 0:
            #if idinc % 1000 == 0:
            #    print(idinc, len(celldict), len(x1), len(x2))
            found = True
            clist = [x1[0], x2[0]]
            x1 = np.delete(x1, 0)
            x2 = np.delete(x2, 0)
            while found:
                res = np.where(x1 == clist[-1])
                if len(res[0]) == 0:
                    found = False
                    continue
                res = res[0][0]
                clist.append(x2[res])
                x1 = np.delete(x1, res)
                x2 = np.delete(x2, res)
            celldict[idinc] = clist
            
            idinc += 1

        self.celldict = celldict