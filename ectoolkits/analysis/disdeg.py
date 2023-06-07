import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List,Optional

from MDAnalysis.lib.distances import capped_distance
from MDAnalysis.analysis.base import AnalysisBase



def count_Oidxs_bonds(positions, o_idxs:List[int], conf_idxs:List[int], box:List[float], 
                      max_cutoff:float=1.2, min_cutoff:Optional[float]=None):
    
    _res = capped_distance(positions[o_idxs], 
                           positions[conf_idxs], 
                           box=box, min_cutoff=min_cutoff, max_cutoff=max_cutoff,
                          return_distances=False)
    #print(_res)
    _res = _res[:,0]
    
    # _res dont care the H idx
    # conf_idxs = 
    #TODO: save the H idxs?
    return np.bincount(_res, minlength=len(o_idxs))

def _count_disdeg(arr):
    return np.count_nonzero(arr < 2)/len(arr)
    
def _count_percent_o(arr):
    return np.count_nonzero(arr == 0)/len(arr)

def _count_percent_oh(arr):
    return np.count_nonzero(arr == 1)/len(arr)

def _count_percent_h2o(arr):
    return np.count_nonzero(arr == 2)/len(arr)

def _count_percent_h3o(arr):
    return np.count_nonzero(arr == 3)/len(arr)

def cumsum_arr(arr):
    #not vectorize:
    #seem same in ectoolkits utils
    cumulative_sum = np.cumsum(arr)
    for i,_num in enumerate(cumulative_sum):
        cumulative_sum[i] =cumulative_sum[i]/(i+1)
    return cumulative_sum

class DisDeg(AnalysisBase):
    
    
    def __init__(self,
                 atomgroup,
                 terminal_oxygen_idxs,
                 chosen_elements=["H"],
                 cellpar=None,
                 cutoff=1.2,
                 min_cutoff=None,):
        
        self._ag = atomgroup
        self._trajectory = self._ag.universe.trajectory
        
        _dimensions = atomgroup.universe.dimensions
        if not (np.any(_dimensions) or np.any(cellpar)):
            raise ValueError("Need cell info!")
        if np.any(_dimensions):
            self.cellpar = _dimensions
        if np.any(cellpar):
            self.cellpar = cellpar
        
        self.terminal_o_idxs = terminal_oxygen_idxs
        
        self.bond_cutoff = cutoff
        self.min_cutoff = min_cutoff
        
        self.chosen_elements = chosen_elements
        
        _elements = self._ag.atoms.elements
        if type(chosen_elements) == str:
            chosen_element = [chosen_elements]
        sel = np.empty(len(_elements), dtype=bool)
        sel.fill(0)
        for element in chosen_elements:
            sel = (sel) | (_elements == element)
        self.conf_idxs = np.where(sel)
        
        #self.disdeg_settings = {}
        
        pass
    
    def _prepare(self): 
        
        onums = len(self.terminal_o_idxs)
        nframes = self.n_frames
        
        self._disdeg = np.empty([0, onums], dtype=int)
        #
    
    
    def _single_frame(self):
        #print(self.cellpar)
        pos = self._ag.positions
        iframe = self._ts.frame
        _idisdeg = count_Oidxs_bonds(pos, 
                                     self.terminal_o_idxs, 
                                     self.conf_idxs, 
                                     self.cellpar,
                                     max_cutoff=self.bond_cutoff,
                                     min_cutoff=self.min_cutoff)
        self._disdeg = np.concatenate((self._disdeg, [_idisdeg]),axis=0)
    
    def _conclude(self):
        
        #TODO: to vectorzie it
        # np.vectoriz
        _arrs = []
        for arr in self._disdeg:
            _arrs.append(_count_disdeg(arr))
        self.disdeg = np.array(_arrs)
        
        _arrs = []
        for arr in self._disdeg:
            _arrs.append(_count_percent_o(arr))
        self.percent_o = np.array(_arrs)
        
        _arrs = []
        for arr in self._disdeg:
            _arrs.append(_count_percent_oh(arr))
        self.percent_oh = np.array(_arrs)
        
        _arrs = []
        for arr in self._disdeg:
            _arrs.append(_count_percent_h2o(arr))
        self.percent_h2o = np.array(_arrs)
        
        _arrs = []
        for arr in self._disdeg:
            _arrs.append(_count_percent_h3o(arr))
        self.percent_h3o = np.array(_arrs)
    
        self.cumave_disdeg = cumsum_arr(self.disdeg)
        self.cumave_percent_o = cumsum_arr(self.percent_o)
        self.cumave_percent_oh = cumsum_arr(self.percent_oh)
        self.cumave_percent_h2o = cumsum_arr(self.percent_h2o)
        self.cumave_percent_h3o = cumsum_arr(self.percent_h3o)
        #print("finished!")
    
    
    def dump_result(self, selects = ["disdeg","percent_o", "percent_oh", "percent_h2o", "percent_h3o"],
                    directory_output="./", plot=True):
        #rawdata
        np.savetxt("raw_disdeg.dat", self._disdeg, fmt="%d")
        print("raw_disdeg.dat saved.")
        
        for _data in selects:
            fn = f"{_data}.dat"
            fn_cumave = f"cumave_{_data}.dat"
            _dat = getattr(self, _data)
            _dat_cumave = getattr(self, "cumave_"+_data)
            np.savetxt(fn, _dat)
            np.savetxt(fn_cumave, _dat_cumave)
            print(fn+" saved.")
            print(fn_cumave+" saved.")
        pass
    
    
    def show_result(self):
        
        fig, ax = plt.subplots(figsize=(12,3),dpi=500)
        sns.lineplot(self.cumave_disdeg, ax=ax, label="disdeg")
        sns.lineplot(self.cumave_percent_o, ax=ax, label="O")
        sns.lineplot(self.cumave_percent_oh, ax=ax, label="OH")
        sns.lineplot(self.cumave_percent_h2o, ax=ax, label="H2O")
        sns.lineplot(self.cumave_percent_h3o, ax=ax, label="H3O")
        
        return fig
    
    def save_obj(self, filename="disdeg.pkl", path_output="./"):
        
        import pickle
        with open(os.path.join(path_output, filename), "wb") as f:
            pickle.dump(self,f)