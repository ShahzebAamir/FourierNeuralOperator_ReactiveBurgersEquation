import re
from os.path import join

import h5py
import numpy as np

from .result_handler import Storage, AlreadyComputedError, Config

class Storage_hdf5_D(Storage):
    
    def __init__(self, alpha, beta=0.1, L=10, N=200,
                 amp=None, wn=None, sign=None, further=True, tmax=None):
        super().__init__(alpha, beta, L, N, amp, wn, sign, tmax, further)
        if tmax:  # if we want to save results
            with h5py.File(self.path_to_file) as file:
                gr = file.require_group(self.group_name)
                if self.dset_name in gr.keys():
                    dset = gr[self.dset_name]
                    if not further:  # refresh data in dataset
                        dset.attrs.clear()
                else:
                    dset = gr.create_dataset(
                        self.dset_name, shape=(0, 3), maxshape=(None, 3),
                        dtype='f8', compression='lzf', fletcher32=True)

                # check if calculation is needed
                tmax_ = dset.attrs.get('tmax', 0.)
                if tmax_ >= tmax:
                    raise AlreadyComputedError(f'Босс, уже посчитано аж для {tmax_}.')

    @property
    def _data_name(self):
        wn = 0.0 if not self.wn else round(self.wn, 3)
        return str(wn)


    @property
    def path_to_file(self):
        return join(Config.path,
                    f'alpha_{self.alpha}beta_{self.beta}',
                    f'alpha_{self.alpha}beta_{self.beta}_D.hdf5')


    @property
    def group_name(self):
        return f'L_{self.L}N_{self.N}/amp_{self.amp}/'

    
    @property
    def dset_name(self):
        return self._data_name + ('' if self.further else 'v2')


    def get_initial_vars(self, **par):
        with h5py.File(self.path_to_file) as file:
            dset = file[self.group_name + self.dset_name]
            if dset.attrs.get('t', 0) and self.further: 
                t = dset.attrs['t']
                dt = dset.attrs['dt']
                x_lab = dset.attrs.get('x_lab', None)
                u = dset.attrs['u']
                return dict(t=t, dt=dt, x_lab=x_lab, u=u)
            else:
                iv = super().get_initial_vars(**par)
                iv['x_lab'] = 0.
                return iv


    def write_data(self, res_t, res_D, res_x_lab, **par):
        with h5py.File(self.path_to_file) as file:
            dset = file[self.group_name + self.dset_name]
            old_len = dset.len()
            new_len = old_len + len(res_t)
            dset.resize(new_len, axis=0)
            dset[old_len:, 0] = res_t
            dset[old_len:, 1] = res_D
            dset[old_len:, 2] = res_x_lab
            if self.further:
                dset.attrs.modify('t', par['t'])
                dset.attrs.modify('dt', par['dt'])
                if par['x_lab'] is not None:
                    dset.attrs.modify('x_lab', par['x_lab'])
                dset.attrs.modify('u', par['u'])
            dset.attrs.modify('tmax', self.tmax)


    def load_tus(self, t_min=0, t_max=np.inf):
        try:
            with h5py.File(self.path_to_file, 'r') as file:
                tus = file[self.group_name + self.dset_name][:]
        except (OSError, KeyError):
            print('С такими параметрами ничего не посчитано, хозяин :(')
            return
        return self._load_tus_bounded(tus, t_min, t_max)

    def __str__(self):
        if self.tmax:
            tmax = self.tmax
        else:
            with h5py.File(self.path_to_file) as file:
                dset = file[self.group_name + self.dset_name]
                tmax = dset.attrs.get('tmax')
        return ''.join(['(hdf5) ', super().__str__(), self.group_name, self.dset_name, f'_tmax_{tmax}'])
