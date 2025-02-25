import numpy as np
import copy

from .frame import Frame
from ..utils import file_utils

class TimeSerie:

    simulation = None
    name = None
    time_window = None
    polyorder = None
    polytype = None
    normalize = None
    fourier_y = None
    frames = []
    time = []
    verbose = False

    def __init__(self, simulation, name, time_window = [], time_frames = [], load=False, fourier_y=False,
                 polyorder=1, polytype='ms', normalize=True, cut_dir=None, cut_coord=None, verbose = False):
        self.simulation = simulation
        self.name = name
        self.time_window = time_window
        self.time_frames = time_frames
        self.polyorder = polyorder
        self.polytype = polytype
        self.normalize = normalize
        self.fourier_y = fourier_y
        self.cut_dir = cut_dir
        self.cut_coord = cut_coord
        self.verbose = verbose
        
        self.frames = []
        self.time = []

        subname = self.simulation.normalization.dict[name+'compo'][0]
        self.filename = self.simulation.data_param.data_files_dict[subname + 'file']
        if load: self.load()

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
            frame = Frame(self.simulation,self.name,tf,load=True, fourier_y=self.fourier_y)
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

    def get_values(self):
        '''
        Get the values of the time serie
        '''
        values = []
        for frame in self.frames:
            values.append(np.squeeze(frame.values))
        return copy.deepcopy(self.time), copy.deepcopy(values)