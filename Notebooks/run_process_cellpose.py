#!/usr/bin/env python

import sys
sys.path.append("../")

from analysis import process_withblue_mean

folder = '/path/to/images'
modelfile = "../Models/cellpose_pombe_transmitted.model"
pkl = "output_data.pkl"

process_withblue_mean.run(folder, pkl, modelfile,
            just_green_min_size=0, green_in_red_min_size=0,
            probability=0.5, n_jobs=1,
            globpattern='*.nd2', zstart=1, zstop=None)

