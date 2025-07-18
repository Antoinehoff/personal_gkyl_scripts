import numpy as np
from ..classes import Frame, TimeSerie

def get_2D_movie_time_serie(simulation,cut_dir,cut_coord,time_frames, fieldnames, fluctuation, fourier_y):
    # Load all data for the movie e.g. [[n0,u0,...],[n1,u1,...],...]
    movie_frames = [[None for _ in range(len(fieldnames))] for _ in range(len(time_frames))]
    fmin = np.zeros_like(fieldnames, dtype=float)
    fmax = np.zeros_like(fieldnames, dtype=float)
    absfmax = np.zeros_like(fieldnames, dtype=float)
    
    for kf,field in enumerate(fieldnames):
        serie = TimeSerie(simulation=simulation, fieldname=field, time_frames=time_frames, load=True, fourier_y=fourier_y)
        serie.slice(cut_dir, cut_coord)
        if len(fluctuation) > 0: serie.fluctuations(fluctuationType=fluctuation)
        # add the frames to the movie
        for it in range(len(time_frames)):
            movie_frames[it][kf] = serie.frames[it].copy()

    # compute max min and maxabs for each fields
    for it,tf in enumerate(time_frames):
        for kf,field in enumerate(fieldnames):
            fmin[kf] = min(fmin[kf], np.min(movie_frames[it][kf].values))
            fmax[kf] = max(fmax[kf], np.max(movie_frames[it][kf].values))
            absfmax[kf] = max(absfmax[kf], np.max(np.abs(movie_frames[it][kf].values)))
    vlims = [[fmin[kf],fmax[kf]] if fmin[kf] >= 0 else [-absfmax[kf],absfmax[kf]] for kf in range(len(fieldnames))]
    return movie_frames, vlims

def get_minmax_values(field = None, simulation=None, fieldname=None, time_frames=None):
    if field is not None:
        fmin = np.min(field)
        fmax = np.max(field)
        nx_mid = field.shape[0]//2
        nz_mid = field.shape[2]//2
        f_OBMP_SOL_min = np.min(field[nx_mid:,:,nz_mid])
        f_OBMP_SOL_max = np.max(field[nx_mid:,:,nz_mid])
        return [fmin, fmax], [f_OBMP_SOL_min, f_OBMP_SOL_max]
    else:
        for tf in time_frames:
            frame = Frame(simulation, fieldname, tf, load=True)
            fmin = np.min(frame.values) if tf == time_frames[0] else min(fmin, np.min(frame.values))
            fmax = np.max(frame.values) if tf == time_frames[0] else max(fmax, np.max(frame.values))
            nx_mid = frame.values.shape[0]//2
            nz_mid = frame.values.shape[2]//2
            f_OBMP_SOL = frame.values[nx_mid:,:,nz_mid]
            f_OBMP_SOL_min = np.min(f_OBMP_SOL) if tf == time_frames[0] else min(f_OBMP_SOL_min, np.min(f_OBMP_SOL))
            f_OBMP_SOL_max = np.max(f_OBMP_SOL) if tf == time_frames[0] else max(f_OBMP_SOL_max, np.max(f_OBMP_SOL))
    return [fmin, fmax], [f_OBMP_SOL_min, f_OBMP_SOL_max]

def get_1xt_diagram(simulation, fieldname, cutdirection, ccoords,
                    tfs):
    tfs = [tfs] if not isinstance(tfs,list) else tfs
    fourier_y = cutdirection.find('k') > -1
    cutdirection = cutdirection.replace('k','')
    # to store times and values
    # get number of time frames
    nt = len(tfs)
    # get number of values
    f0 = Frame(simulation,fieldname,tfs[0],load=True, fourier_y=fourier_y)
    f0.slice(cutdirection,ccoords)
    # get number of values
    nv = np.squeeze(f0.values.size)
    t  = np.zeros(nt)
    values = np.zeros((nv,nt))
    # Fill ZZ with data for each time frame
    for it,tf in enumerate(tfs):
        frame = Frame(simulation,fieldname,tf,load=True, fourier_y=fourier_y)
        frame.slice(cutdirection,ccoords)
        t[it] = frame.time
        values[:,it] = np.squeeze(frame.values)
    if values.ndim == 1: values = values.reshape(1,-1)
    x = frame.new_grids[0]
    tsymb = simulation.normalization.dict['tsymbol'] 
    tunit = simulation.normalization.dict['tunits']
    tlabel = tsymb+(' ('+tunit+')')*(1-(tunit==''))
    xlabel = frame.new_gsymbols[0]+(' ('+frame.new_gunits[0]+')')*(1-(frame.new_gunits[0]==''))
    vlabel = frame.vsymbol+(' ('+frame.vunits+')')*(1-(frame.vunits==''))
    slicetitle = frame.slicetitle
    return x,t,values,xlabel,tlabel,vlabel,frame.vunits,slicetitle, fourier_y