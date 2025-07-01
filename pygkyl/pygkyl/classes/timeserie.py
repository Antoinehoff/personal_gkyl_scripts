import numpy as np
import copy

from .frame import Frame
from ..utils import file_utils

class TimeSerie:

    simulation = None
    fieldname = None
    time_window = None
    polyorder = None
    polytype = None
    normalize = None
    fourier_y = None
    frames = []
    time = []
    verbose = False

    def __init__(self, simulation, fieldname, time_frames = [], time_window = [], load=False, fourier_y=False,
                 polyorder=1, polytype='ms', normalize=True, cut_dir=None, cut_coord=None, verbose = False
                 , frames = None):
        self.simulation = simulation
        self.fieldname = fieldname
        self.time_frames = time_frames if isinstance(time_frames, list) else [time_frames]
        self.time_window = time_window if isinstance(time_window, list) else [time_window]
        self.polyorder = polyorder
        self.polytype = polytype
        self.normalize = normalize
        self.fourier_y = fourier_y
        self.cut_dir = cut_dir
        self.cut_coord = cut_coord
        self.verbose = verbose
        
        self.frames = []
        self.time = []

        # subname = self.simulation.normalization.dict[fieldname+'compo'][0]
        subname = self.simulation.data_param.field_info_dict[fieldname+'compo'][0]
        self.filename = self.simulation.data_param.file_info_dict[subname + 'file']
        if load: 
            self.load()
        elif frames is not None: self.init_from_frames(frames)
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.free_values()

    def load(self):
        '''
        Load the time serie
        '''
        # get all the available time frames
        all_tf = file_utils.find_available_frames(self.simulation,dataname=self.filename)
        if self.verbose: print('Available time frames of ',self.filename,' :',all_tf)
        # select only the ones in the time frames if time_frames is provided
        if self.time_frames:
            time_frames = [tf for tf in all_tf if tf in self.time_frames]
        else:
            time_frames = all_tf
        # get the frames
        for it,tf in enumerate(time_frames):
            frame = Frame(self.simulation,self.fieldname,tf,load=True, fourier_y=self.fourier_y)
            add_frame = False
            if self.time_window:
                if frame.time >= self.time_window[0] and frame.time <= self.time_window[1]:
                    add_frame = True
            else:
                add_frame = True
            if add_frame:
                if not self.cut_dir is None:
                    frame.slice(self.cut_dir, self.cut_coord)
                self.frames.append(frame)
                self.time.append(frame.time)
        if len(self.frames) > 0:
            self.vsymbol = self.frames[0].vsymbol
            self.vunits = self.frames[0].vunits
            self.gsymbols = self.frames[0].gsymbols
            self.gunits = self.frames[0].gunits
                  
    def init_from_frames(self, frames):
        '''
        Initialize the time serie from a list of frames
        '''
        self.frames = frames
        self.time = [frame.time for frame in frames]

    def get_values(self):
        '''
        Get the values of the time serie
        '''
        values = []
        for frame in self.frames:
            values.append(np.squeeze(frame.values))
        return copy.deepcopy(self.time), copy.deepcopy(values)
    
    def get_time_average(self):
        v_tavg = self.frames[0].values
        time = self.frames[0].time
        for frame in self.frames[1:]:
            dt = frame.time - time
            v_tavg += frame.values * dt
            time = frame.time
        v_tavg /= time - self.frames[0].time  
        return v_tavg
    
    def get_y_average(self, output_plane='xz', cut_coord=0, fidx=-1):
        '''
        Get the y average of the time serie
        '''
        avg_frame = self.frames[fidx].copy()
        if output_plane == 'xz':
            avg_frame.slice('xz', 'avg')
            mean_values = avg_frame.values
        elif output_plane == 'xy':
            avg_frame.slice('x', ['avg',cut_coord])
            mean_values = np.repeat(avg_frame.values, self.frames[-1].values.shape[1], axis=1)
        elif output_plane == 'yz':
            avg_frame.slice('z', ['avg',cut_coord])
            mean_values = np.repeat(avg_frame.values, self.frames[-1].values.shape[0], axis=0)
        return copy.deepcopy(mean_values)
    
    def slice(self, cut_dir, cut_coord):
        '''
        Slice the time serie
        '''
        for frame in self.frames:
            frame.slice(cut_dir, cut_coord)
            
    def average(self, averageType='tavg', fidx=-1):
        if 'tavg' in averageType:
            favg = self.get_time_average()
            self.vsymbol = r'$\langle$' + self.vsymbol + r'$\rangle_t$'
        elif 'yavg' in averageType:
            favg = self.get_y_average(fidx=fidx)
            self.vsymbol = r'$\langle$' + self.vsymbol + r'$\rangle_y$'
        
        # Update the values of the frames
        for i in range(len(self.frames)):
            self.frames[i].values = 0*self.frames[i].values + favg

    def fluctuations(self,fluctuationType='',avg_array=None):
        if avg_array is None:
            if 'tavg' in fluctuationType:
                tavg = self.get_time_average()
                favg = lambda fidx: tavg
            elif 'yavg' in fluctuationType:
                favg = lambda fidx: self.get_y_average(fidx=fidx)
        else:
            favg = lambda fidx: avg_array
            
        # Compute the fluctuations
        for i in range(len(self.frames)):
            avg = favg(i)
            self.frames[i].values = self.frames[i].values - avg
            if 'relative' in fluctuationType:
                # Avoid division by zero
                self.frames[i].values = np.where(avg != 0.0, 100.0 * self.frames[i].values / avg, 0.0)
                
        self.vsymbol = self.vsymbol + r' $-\langle$'+self.vsymbol+r'$\rangle_%s$'%('t' if 'tavg' in fluctuationType else 'y')
            
        if 'relative' in fluctuationType:
            self.vsymbol = r'(' + self.vsymbol + r') $/\langle$'+self.frames[0].vsymbol+r'$\rangle_y$'
            self.vunits = r'%'
          
    def free_values(self):
        '''
        Free the values of the time serie
        '''
        for frame in self.frames:
            frame.free_values()
        self.frames = []
        self.time = []
        self.values = None