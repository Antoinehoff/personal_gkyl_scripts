import numpy as np
from ..classes import Frame

def get_2D_movie_data(simulation,cut_dir,cut_coord,time_frames, fieldnames, fluctuation):
    # Load all data for the movie e.g. [[n0,u0,...],[n1,u1,...],...]
    movie_frames = []
    fmin = np.zeros_like(fieldnames, dtype=float)
    fmax = np.zeros_like(fieldnames, dtype=float)
    absfmax = np.zeros_like(fieldnames, dtype=float)

    for tf in time_frames:
        fields_t = []
        for field in fieldnames:
            frame = Frame(simulation, field, tf, load=True)
            frame.slice_2D(cut_dir, cut_coord)
            fields_t.append(frame)
        movie_frames.append(fields_t)
    if fluctuation:
        # compute the time average for each fields
        for kf,field in enumerate(fieldnames):
            tavg_field = 0
            t0 = movie_frames[0][kf].time
            t  = t0
            for it in range(len(time_frames)-1):
                dt = movie_frames[it+1][kf].time - t
                tavg_field += movie_frames[it+1][kf].values * dt
                t += dt
            tavg_field /= (t-t0)
            # substract the average over time for each field
            for it, tf in enumerate(time_frames):
                movie_frames[it][kf].values -= tavg_field
                if fluctuation == "relative":
                    movie_frames[it][kf].values *= 100.0/tavg_field

    # compute max min and maxabs for each fields
    for it,tf in enumerate(time_frames):
        for kf,field in enumerate(fieldnames):
            fmin[kf] = min(fmin[kf], np.min(movie_frames[it][kf].values))
            fmax[kf] = max(fmax[kf], np.max(movie_frames[it][kf].values))
            absfmax[kf] = max(absfmax[kf], np.max(np.abs(movie_frames[it][kf].values)))
    vlims = [[fmin[kf],fmax[kf]] if fmin[kf] >= 0 else [-absfmax[kf],absfmax[kf]] for kf in range(len(fieldnames))]
    return movie_frames, vlims

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