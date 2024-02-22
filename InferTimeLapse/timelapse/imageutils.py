from re import I
import tifffile
import numpy as np
from skimage.measure import label
import av


def label_mask(x, threshold=0.85):
    
    p = (x > threshold).astype(np.uint8)
    labeled_list = [label(j) for j in p]
    labeled = np.stack(labeled_list)
    return labeled

def make_movie(snaps, outfile, fps):
    
    height, width, _ = snaps[0].shape 
    container = av.open(outfile, mode='w')
    stream = container.add_stream('h264', rate=fps)  
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuv420p'     
    
    for i, s in enumerate(snaps):
        vf = av.VideoFrame.from_ndarray(s[:,:,:3], format='rgb24')
        for packet in stream.encode(vf):
            container.mux(packet) 
    
    for packet in stream.encode():
        container.mux(packet) 
    container.close() 