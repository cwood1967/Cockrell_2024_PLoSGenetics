
import sys
sys.path.append("../")

from analysis import screen

folders = ["path/to/phenix/screening/Images"]

modelfile = ".../Models/torch_maskrcnn_screen.pt"

pkl = "data_output.pkl"
screen.run_screen(folders, pkl, modelfile,
            just_green_min_size=0, green_in_red_min_size=0,
           globpattern='*.tif', channels=[1,0,2])



