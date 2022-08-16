#!/usr/bin/env python
# coding: utf-8

import os
from os.path import join, dirname, basename
from abc import ABC, abstractmethod
import re
import numpy as np
import h5py
import matplotlib.pyplot as plt


_path = join('..', 'comp_res', 'prA')


class Config:
    path = _path


class AlreadyComputedError(ValueError):
    '''Raised to prevent computation when the task with passed parameters has already been computed.'''

#############################################################################

class Storage(ABC):
    
    def __init__(self, alpha, beta, L, N, amp, wn, sign, tmax, further):
        self.alpha = alpha
        self.beta = beta
        self.L = L
        self.N = N
        self.amp = amp if amp else 0.0
        self.wn = (wn if wn else None) if amp else None
        self.sign = (sign if sign else 1) if wn else None
        if tmax == 0:
            raise ValueError('`tmax` must be greater then 0.')
        self.tmax = tmax
        self.further = further
        
    @staticmethod
    def parse(param, type_, s):
        if type_ == int:
            number_group = r'(\d+)'
        elif type_ == float:
            number_group = r'(\d+(?:\.?\d+))'
        regexp = param + '_' + number_group
        m = re.search(regexp, s)
        return type_(m[1]) if m else None
    
    def set(self, **pars):
        # setting tmax to None tells to switch from Storage
        # that saves data to one that loads data
        if (not self.tmax or 
            ('tmax' in pars and not pars['tmax'])):
            if 'tmax' in pars:
                self.tmax = None
            if 'alpha' in pars:
                self.alpha = pars['alpha']
            if 'beta' in pars:
                self.beta = pars['beta']
            if 'L' in pars:
                self.L = pars['L']
            if 'N' in pars:
                self.N = pars['N']
            if 'amp' in pars:
                self.amp = pars['amp']
            if 'wn' in pars:
                self.wn = pars['wn']
            if 'sign' in pars:
                self.sign = pars['sign']
            if 'further' in pars:
                self.further = pars['further']
        else:
            raise ValueError('Parameters can be set only when tmax is None.')


    def get_initial_vars(self, **par):
        t = 0
        dt = par['dt']
        x_lab = 0. if self.wn else None
        u = par['u']
        return dict(t=t, dt=dt, x_lab=x_lab, u=u)
    

    @abstractmethod
    def write_data(self, t, us, **par):
        pass

    @property
    def _data_name(self):
        if self.wn and self.amp:
            sgn = 'm' if self.sign == -1 else 'p'
            data_name = f'amp_{self.amp}wn_{self.wn}{sgn}'
        else:
            data_name = f'amp_{self.amp}'
        return data_name

    @abstractmethod
    def load_tus(self, t_min, t_max):
        pass

    def _load_tus_bounded(self, tus, t_min, t_max):
        if t_min == 0 and t_max == np.inf:
            return tus
        else:
            idx = np.logical_and(tus[:, 0] >= t_min, tus[:, 0] <= t_max)
            return tus[idx]

        
    def plot(self, t_min=0, t_max = np.inf):
        data = self.load_tus(t_min, t_max)
        plt.plot(data[:, 0], data[:, 1])
        plt.gcf().set_size_inches((12, 6))
        plt.title(self)
        plt.grid()
        plt.show()

    def __str__(self):
        return f'alpha_{self.alpha}beta_{self.beta}/'

#############################################################################

class Storage_txt(Storage):
    
    def __init__(self, alpha, beta=0.1, L=10, N=200,
                 amp=None, wn=None, sign=None, further=True, tmax=None):
        '''if tmax is None, we want to find file with max available tmax'''
        super().__init__(alpha, beta, L, N, amp, wn, sign, tmax, further)
        if tmax:
            os.makedirs(self.path, exist_ok=True)

    @classmethod
    def parse_params(cls, path_to_file):
        '''parse parameters from the path_to_file, both
        continue and data'''

        par = {}
        par['alpha'] = cls.parse('alpha', float, path_to_file)
        par['beta'] = cls.parse('beta', float, path_to_file)
        par['L'] = cls.parse('L', int, path_to_file)
        par['N'] = cls.parse('N', int, path_to_file)
        par['amp'] = cls.parse('amp', float, path_to_file)
        wn_sign = re.search(r'wn_(\d+(?:\.?\d+))([pm])', path_to_file)
        if wn_sign:
            par['wn'] = float(wn_sign[1])
            sgn = wn_sign[2]
            par['sign'] = 1 if sgn == 'p' else -1
        par['tmax'] = cls.parse('tmax', int, path_to_file)
        par['further'] = False if 'v2' in path_to_file else True
        return par


    @property
    def path(self):
        return join(Config.path,
                    f'alpha_{self.alpha}beta_{self.beta}', f'L_{self.L}N_{self.N}')


    @property
    def _max_tmax(self):
        try:
            files = os.listdir(self.path)
        except FileNotFoundError:
            return 0
        # find res file with maximal available tmax
        tmaxx = 0
        pattern = ''.join([self._data_name,
                           '_' if self.wn and self.amp else '',
                           'tmax_(\d+)',
                           ('' if self.further else 'v2'),
                           '.txt'])
        for f in files:
            m = re.match(pattern, f, flags=re.ASCII)
            if m and int(m.group(1)) > tmaxx:
                tmaxx = int(m.group(1))
        return tmaxx


    @property
    def files(self):
        underscore = '_' if self.wn and self.amp else ''
        suffix = ('' if self.further else 'v2') + '.txt'
        res_template = ''.join([self._data_name, underscore, 'tmax_{}', suffix])
        continu_template = (f'amp_{self.amp}wn_{self.wn}{"p" if self.sign == 1 else "m"}.continue' 
                    if self.wn and self.amp else f'amp_{self.amp}.continue') + '{}.txt'
        # return pattern for searching tmax if _tmax is not initialised
        max_tmax = self._max_tmax
        if self.further:
            if not (self.tmax or max_tmax):
                raise FileNotFoundError('С такими параметрами ничего не посчитано, хозяин :(')
            elif not self.tmax and max_tmax:
                t_res, m_res, t_cont, m_cont = max_tmax, 'r', max_tmax, 'r'
            elif self.tmax and not max_tmax:
                t_res, m_res, t_cont, m_cont = self.tmax, 'w', self.tmax, 'w'
            elif self.tmax > max_tmax:
                t_res, m_res, t_cont, m_cont = max_tmax, 'a', max_tmax, 'r+'
            elif self.tmax <= max_tmax:
                raise AlreadyComputedError(f'Босс, уже посчитано аж для {max_tmax}.')
        else:
            if self.tmax:
                t_res, m_res, t_cont, m_cont = self.tmax, 'w', None, None
            elif not self.tmax and max_tmax:
                t_res, m_res, t_cont, m_cont = max_tmax, 'w', None, None
            else:
                raise FileNotFoundError('С такими параметрами ничего не посчитано, хозяин :(')
        return {
            'res': {
                'file': join(self.path, res_template.format(t_res)),
                'mode': m_res
            },
            'continu': {
                'file': join(self.path, continu_template.format(t_cont)),
                'mode': m_cont
            } if m_cont and t_cont else None
        }
                

    def get_initial_vars(self, **par):
        continu_ = self.files['continu']
        if continu_ and continu_['mode'] in ('r', 'r+'):
            with open(**continu_) as continu:
                t = float(continu.readline().split('=')[1])
                dt = float(continu.readline().split('=')[1])
                try:
                    x_lab = float(continu.readline().split('=')[1])
                except ValueError:
                    x_lab = None
                u = np.loadtxt(continu)
                return dict(t=t, dt=dt, x_lab=x_lab, u=u)
        else:
            return super().get_initial_vars(**par)


    def _write_metadata(self, res, **par):
        '''Writes parameters such as alpha, beta and so on to the file.'''
        res.write(f'alpha = {self.alpha}\n')
        res.write(f'beta = {self.beta}\n')
        res.write(f'L = {self.L}\n')
        res.write(f'N = {self.N}\n')
        for key, value in par.items():
            if key not in ['t', 'dt', 'x_lab', 'u']:
                res.write("{} = {}\n".format(key, value))
        res.write('\n{:^22s} | {:^22s}\n'.format('t', 'us'))


    def write_data(self, res_t, res_us, **par):
        '''Writes data to the file. If further is True, also writes data from **par to continue file.'''
        files = self.files
        res_ = files['res']
        with open(**res_) as res:
            if res_['mode'] == 'w':
                self._write_metadata(res, **par)
            for i in range(len(res_t)):
                res.write('{:<20.16e} | {:<20.16e}\n'.format(res_t[i], res_us[i]))
        
        if self.further:
            if res_['mode'] == 'a':  # rename file, changing the part with tmax
                os.rename(res_['file'],
                          re.sub(r'(?<=tmax_)\d+', str(self.tmax), res_['file']))
            continu_ = files['continu']
            with open(**continu_) as continu:
                continu.write(f"t = {par['t']}\n")
                continu.write(f"dt = {par['dt']}\n")
                continu.write(f"x_lab = {par['x_lab']}\n")
                np.savetxt(continu, par['u'])
                continu.truncate()
            if continu_['mode'] == 'r+':
                os.rename(continu_['file'],
                          re.sub(r'\d+$', str(self.tmax), continu_['file']))


    def load_tus(self, t_min=0, t_max=np.inf):
        with open(self.files['res']['file']) as file:
            for line in file:
                if line.lstrip().startswith('t '):
                    break
            tus = np.loadtxt(file, delimiter='|')
        return self._load_tus_bounded(tus, t_min, t_max)
                       
    def __str__(self):
        if self.tmax:
            tmax = self.tmax
        else:
            tmax = self.parse('tmax', int, self.files['res']['file'])
        return ('(txt) ' + super().__str__() + f'L_{self.L}N_{self.N}/' +
                re.sub(r'tmax_\d+', f'tmax_{tmax}', basename(self._data_name)))

#############################################################################

class Storage_hdf5(Storage):
    
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
                        self.dset_name, shape=(0, 2), maxshape=(None, 2),
                        dtype='f8', compression='lzf', fletcher32=True)

                # check if calculation is needed
                tmax_ = dset.attrs.get('tmax', 0.)
                if tmax_ >= tmax:
                    raise AlreadyComputedError(f'Босс, уже посчитано аж для {tmax_}.')


    @property
    def path_to_file(self):
        return join(Config.path,
                    f'alpha_{self.alpha}beta_{self.beta}',
                    f'alpha_{self.alpha}beta_{self.beta}.hdf5')


    @property
    def group_name(self):
        return f'L_{self.L}N_{self.N}/'

    
    @property
    def dset_name(self):
        return super()._data_name + ('' if self.further else 'v2')


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
                return super().get_initial_vars(**par)


    def write_data(self, res_t, res_us, **par):
        with h5py.File(self.path_to_file) as file:
            dset = file[self.group_name + self.dset_name]
            old_len = dset.len()
            new_len = old_len + len(res_t)
            dset.resize(new_len, axis=0)
            dset[old_len:, 0] = res_t
            dset[old_len:, 1] = res_us
            for key, value in par.items():
                if value is None:
                    continue
                dset.attrs.modify(key, value)
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

#############################################################################

def get_storage(format='hdf5', **par):
    if format == 'hdf5':
        return Storage_hdf5(**par)
    elif format == 'txt':
        return Storage_txt(**par)
