
import sys
sys.path.append("../")

from analysis import process

folder = '/path/to/images'
modelfile = "../Models/torch_maskrcnn_screen.pt"
pkl = "output_data.pkl"

process.run(folder, pkl, modelfile,
            just_green_min_size=0, green_in_red_min_size=0,
            probability=0.5, n_jobs=2,
            globpattern='*.nd2', zstart=1, zstop=None)



