import os

import pandas as pd
import numpy as np
from tracking import tracker
from pombelength import cell_length
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from . import imageutils

def trackcells(df, frame_col='frame',
               xm='cmx', ym='cmy', max_separation=10, max_lost=2):
    
    t = tracker.Tracker(df, frame_col='frame',
                    xm='cmx', ym='cmy', max_separation=10, max_lost=2)

    t.run()  
    xdf = t.df.copy()
    xdf = xdf.dropna()
    xdf["cellid"] = xdf.cellid.astype('int')
    # xdf.loc[: 'cellid'] = xdf.loc[:, 'cellid'].astype('int')
    zdf = xdf[['cellid', 'frame']].groupby('cellid').transform(lambda x : x.min())
    xdf['rel_frame'] = xdf.frame - zdf.frame
    return xdf

def get_cell_lengths(rdir, labeled):
    pf = f"{rdir}/cell_length.pickle"

    if os.path.exists(pf):
        length_df = pd.read_pickle(pf)
    else:
        length_df = length_dataframe(labeled)
        length_df.to_pickle(pf)
    length_df.frame = length_df.frame.astype(np.int32)

    return length_df

def make_results_dir(imagefilename):
    bext = os.path.basename(imagefilename)
    lastdot = bext.rindex('.')
    b = bext[:lastdot]
   
    rdir =  f'Results/{b}'
    if not os.path.exists(rdir):
        os.makedirs(rdir)

    return rdir
                
def length_dataframe(pdata):
   
    dflist = list()
    erx = "clean"
    for j, labels in enumerate(pdata):
        #print(j)
        nlabels = labels.max()
         
        for i in range(1, nlabels + 1):
        
            mx = np.where(labels == i)
            cmx = mx[1].mean()
            cmy = mx[0].mean()
            area = len(mx[0])
            #try:
            length_dict = cell_length.calc_cell_length(mx)
            if length_dict is None:
                continue
            #lengths.append(length_dict['length'])
            length = length_dict['length']
            dux = length_dict['Xp'].max() - length_dict['Xp'].min()
            duy = length_dict['yp'].max() - length_dict['yp'].min()

#                 xdiff.append(dux)
#                 ydiff.append(duy)

            _df = pd.Series({'length':length,
                    'xdiff':dux,
                    'ydiff':duy,
                    #'xdiff':xdiff,
                    #'ydiff':ydiff,
                    'cmx':cmx,
                    'cmy':cmy,
                    'area':area,
                    'e0':length_dict['eigenvec'][0],
                    'e1':length_dict['eigenvec'][1],
                    })
            _df['frame'] = j
            _df['label'] = i
            dflist.append(_df)
#             except Exception as e:
#                 print(e)
#                 continue
    
    length_df = pd.concat(dflist, axis=1).T
    return length_df

    
def cellid_row_table(df, xdf, labeled):
    cnt = 0
    cell_dict = dict()
    for row in df.itertuples():
        x = row.xcm
        y = row.ycm
        t = row.frame
        roi = labeled[t, int(y), int(x)]        
        xdf_row = xdf[(xdf.frame == t) & (xdf.label == roi)]
        if len(xdf_row) > 0:
            a = row.Index
            b = xdf_row.index.values[0]
            cellid =  xdf_row.cellid.values[0]
            if cellid in cell_dict:
                cell_dict[cellid]['f'].append(a)
                cell_dict[cellid]['length'].append(b)
            else:
                cell_dict[cellid] = {'f':[a], 'length':[b]}   

    return cell_dict

    
def readcell(cellid, df, xdf, cell_dict):
    cellrows = cell_dict[cellid]
    
    cols = ['green_mean', 'rdna_mean', 'red_mean',
           'vol', 'rdna_vol', 'red_voxels',
           'xcm', 'ycm', 'frame',
           'green_rdna_dist', 'green_red_dist', 'rdna_red_dist']
    
    _vf = df.loc[cellrows['f']][cols].copy().reset_index()
    _vf= _vf.rename(columns={'index':'f_index'})
    _v = xdf.loc[cellrows['length']][['length', 'rel_frame', 'frame', 'label']].copy().reset_index()
    _v = _v.rename(columns={'index':'len_index'})
    _v = _v.merge(_vf, left_index=True, right_index=True, suffixes=[None, '_f'])
    _v['dv'] = np.sqrt(_v.xcm**2 + _v.ycm**2)
    _v = _v.sort_values(['rel_frame', 'dv'])
    _v['cellid'] = cellid
    _v['cc'] = 1
    _v['cc'] = _v.cc.astype(np.int32)
    _v['nucnum'] = _v.groupby('rel_frame').transform(lambda x : x.cumsum()).cc
   
    _v['rdna_green_ratio'] = _v.rdna_mean/_v.green_mean
    _v['rdna_vol_ratio'] = _v.rdna_vol/_v.vol

    return _v

    
def plot_trace(cellid, df, xdf, cell_dict, v1, v2=None):
    res = readcell(cellid, df, xdf, cell_dict)
    
    if v2 is None:
        nplots = 2
    else:
        nplots = 3
        
    fig, ax = plt.subplots(nplots, 1, figsize = (8,6))
    
    _d = res.loc[res.nucnum == 1]
    ax[0].plot(_d['frame'], _d[v1])
    ax[nplots - 1].plot(_d['frame'], _d['length'])
                
    _d = res.loc[res.nucnum == 2]
    ax[0].plot(_d['frame'], _d[v1])

    if v2 is not None:
        res2 = readcell(cellid, df, xdf, cell_dict)
        
        _d = res2.loc[res2.nucnum == 1]
        ax[1].plot(_d['frame'], _d[v2])
        _d = res2.loc[res2.nucnum == 2]
        ax[1].plot(_d['frame'], _d[v2])

def cell_by_nuc(df, xdf, cell_dict, labeled, nucx, nucy, frame):
    clabel = labeled[frame, nucy, nucx]
    cid = xdf[(xdf.label == clabel) & (xdf.frame == frame)].cellid.values[0]
    #print(cid)
    _df = readcell(cid, df, xdf, cell_dict)
    return cid, _df

    
def find_cell(df, xdf, labeled, y, x, frame):
    last_yx = np.array([[y, x]])
    next_yxs = df.loc[df.frame == frame][['ycm', 'xcm']].values
    if len(next_yxs) < 1:
        return -1
    cds = cdist(last_yx, next_yxs)
    next_index = cds.argmin()
    idx = next_yxs[next_index]
    for i in range(4):
        xframe =  frame + i
        if xframe >= len(labeled):
            break
        next_label = labeled[xframe, idx[0], idx[1]]
        _df = xdf[(xdf.label == next_label) & (xdf.frame == xframe)]
            
        _next_cellid = xdf[(xdf.label == next_label) & (xdf.frame == xframe)].cellid#.values[0]
        if len(_next_cellid) > 0:
            break
    
    if len(_next_cellid) > 0:
        next_cellid = _next_cellid.values[0]
    else:
        next_cellid = -1
        #print("not found")
    return next_cellid
    
def find_next_cells(df, xdf, labeled, cell_dict, cellid, rxdf, trace_dict):
    last_frame = rxdf.loc[rxdf.frame == rxdf.frame.max()]
    found_cells = list()
    for f in last_frame.itertuples():
        next_frame = int(f.frame + 1)
        next_cellid = find_cell(df, xdf, labeled,
                                f.ycm, f.xcm, int(f.frame + 1))
        if next_cellid < 0:
            continue
        found_cells.append(next_cellid)
        trace_dict[next_cellid] = readcell(next_cellid, df, xdf, cell_dict)
    return found_cells

    
def plot_cell_trace(trace_df, q, max_frame=None, ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(8,6)) 
    
    if max_frame is None:
        max_frame = trace_df.frame.max()
    if max_frame == 0:
        return
    ymax = trace_df[q].max()
    for c in trace_df.nucnum.unique():
        _lax = sns.lineplot(data=trace_df[(trace_df.frame <= max_frame + 0)
                                   & (trace_df.nucnum == c)],
                        x='frame', y=q,
                        hue='cellid', palette="tab10", ax=ax)
        
        palt = _lax.get_lines()
        for line in palt:
            verts = line.get_path().vertices
            if len(verts) < 1:
                continue
            lastx = verts[-1, 0]
            lasty = verts[-1, 1]
            if np.abs(lastx) == np.inf:
                continue
            lastx = int(lastx)
            if lastx == max_frame:
                _lax.scatter(lastx, lasty, color=line.get_color())

        
    ax.set_xlim(0, trace_df.frame.max()+3)
    ax.set_ylim(0, 1.2*ymax)
    ax.legend().remove()

    
def normpatch(data, channel, sy, sx):
    _p = data[:, channel, sy, sx]
    _p = (_p - _p.min())/(_p.max() - _p.min())
    if channel == 2:
        for i, p2 in enumerate(_p):
            #print("A", p2.mean(), p2.std(), p2.min(), p2.max())
            p2 = (p2 - p2.min())/(p2.max() - p2.min())
            p2 = p2 - p2.mean()
            p2 = p2/p2.std()
            #print("B", p2.mean(), p2.std(), p2.min(), p2.max())
            p2 = 128 + 8*p2
            #print("C", p2.mean(), p2.std())
            p2 = np.where(p2 > 255, 255, p2)
            p2 = np.where(p2 < 0, 0, p2)
            #print("D", p2.mean(), p2.std(), p2.min(), p2.max())
            #print('--')
            _p[i] = p2
    else:
        _p = 128*_p
    return _p.astype(np.uint8)

    
def get_rgb_patch(data, cx, cy):
    d = 128
    c0y = max(cy - d, 0)
    c0x = max(cx - d, 0)
    cfy = min(cy + d, data.shape[-2])
    cfx = min(cx + d, data.shape[-1])
    sy = slice(c0y, cfy)
    sx = slice(c0x, cfx)

    c1 = normpatch(data, 0, sy, sx)
    c2 = normpatch(data, 1, sy, sx)
    c3 = normpatch(data, 2, sy, sx)
    
    
    r = c1 + c3
    g = c2 + c3
    b = c1 + c3
    r = np.where(r > 255, 255, r)
    g = np.where(g > 255, 255, g)
    b = np.where(b > 255, 255, b)
    
    patch = np.stack([r,g,b], axis=-1)
    return patch

def to_rgb(images):
 
    xlist = list()
    for _x in images:
        xlist.append((_x - _x.min())/(_x.max() - _x.min()))

    r = xlist[0] + xlist[2]
    g = xlist[1] + xlist[2]
    rgb = np.stack([r, g, r], axis=-1)
    rgb = np.where(rgb > 1, 1, rgb)
    return rgb
    
def plot_timelapse(trace_df, data, cellx, celly, q, rdir):
    patch = get_rgb_patch(data, cellx, celly)

    outfile = f"{rdir}/{cellx:04d}-{celly:04d}-{q}.mp4"
    vidx = list()
    for i, p in enumerate(patch):
        if i == 0:
            continue
        _fx = Figure(constrained_layout=False, figsize=(12,4))
        canvas =FigureCanvas(_fx)
        gs = _fx.add_gridspec(1, 3)
        ax0 = _fx.add_subplot(gs[0, 2])
        ax0.axis('off')
        ax0.imshow(p)
        ax1 = _fx.add_subplot(gs[0, :2])
        plot_cell_trace(trace_df, q, max_frame=i, ax=ax1)
        canvas.draw()
        buf = canvas.buffer_rgba()
        vidx.append(np.asarray(buf))
        plt.close(_fx)

    imageutils.make_movie(vidx, outfile, 8)
        #if i > 2:break    
    return vidx

    
def track_cell(df, xdf, cell_dict, labeled, cellx, celly, frame):
    cellid, rx = cell_by_nuc(df, xdf, cell_dict, labeled, cellx, celly, frame)
    trace_dict = {cellid:rx}
    cids = [cellid]
    while len(cids) > 0:
        found_cells = find_next_cells(df, xdf, labeled, cell_dict,
                                            cids[0],
                                            trace_dict[cids[0]],
                                            trace_dict)
        cids.extend(found_cells)
        cids.pop(0)
    
    trace_df = pd.concat(list(trace_dict.values()))
    trace_df['cellid_nuc'] = trace_df.cellid.astype(str) + '-' + trace_df.nucnum.astype(str)
    trace_df = trace_df.reset_index()

    return cellid, trace_df
        