import os

from joblib import Parallel, delayed
import pandas as pd

from nd2reader import ND2Reader
import tifffile
import zarr

import process_timelapse as process

class timelapse:
    
    def __init__(self, filename, channels, cnn=None, is_cellpose=None):
        self.filename = filename
        self.filetype = self.filename[-3:]
        self.channels = channels
        self.just_green_min_size = 0
        self.just_red_min_size = 0
        self.cnn = cnn 
        self.is_cellpose = is_cellpose

        if self.filetype == "nd2":
            nd2 = ND2Reader(filename)
            self.nframes = nd2.sizes['t']
            self.read_frames(nd2) 
            nd2.close()
            del nd2
        else:
            self.store = tifffile.imread(filename, aszarr=True)
            self.z = zarr.open(self.store, mode='r')
            self.read_frames(self.z)
            self.store.close()

    def read_frames(self, _data):
        self.frames = dict()
        if self.filetype == 'nd2':
            for i in range(self.nframes):
                frame = process.getframe(_data, i, channels=self.channels)
                self.frames[i] = frame 
        else:
            for i, _s in enumerate(self.z):
                self.frames[i] = _s 

    # def get_frame(self, frame):
    #     #nd2 = ND2Reader(self.filename)
    #     frame = process.getframe(self.nd2, frame, channels=self.channels)
    #     return frame
   
    def frame_gen(self):
        for i, data in self.frames.items():
            print("in gen", i)
            yield {'frame':i, 'data':data}
        
    def _pfunc(self, d, **kwargs):
        frame = d['frame']
        image = d['data'][1:]
        print(frame, image.shape)
        _, sspdf  = process.process(self.filename, image,
                                    self.cnn, self.is_cellpose,
                                    frame,
                                    self.just_green_min_size,
                                    self.just_red_min_size
        )
        
        sspdf['frame'] = frame
        return sspdf

    def analyze(self):
        if not os.path.exists('Data/Masks'):
            os.makedirs('Data/Masks')
        if not os.path.exists('Data/NoRDNA'):
            os.makedirs('Data/NoRDNA')

#        dflist = Parallel(n_jobs=4)\
#            (delayed(self._pfunc)(d) for d in self.frames)
    
        dflist = list()
        for d in self.frame_gen():
            _df = self._pfunc(d)
            dflist.append(_df)

        res = pd.concat(dflist)
        bn = os.path.basename(self.filename)[:-4]
        res.to_pickle(f"{bn}_timelapse_df.pkl")


if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    ### the cellpose model file
    modelfile = ("../Models/cellpose_pombe_transmitted.model"
    )

    cnn = process.cellpose_setup(modelfile, 50)
    is_cellpose = True
    movie = timelapse(filename, [0,1, 2, 3], cnn=cnn, is_cellpose=True)
    movie.analyze()
