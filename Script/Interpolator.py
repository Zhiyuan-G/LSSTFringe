import astropy.units as u
import numpy as np
import batoid
from LSSTFringe import utils,plot_utils,TMMSIM
index_of_refraction = utils.load_refraction_data(Epoxy_ind=1.6,Temp = 153.)
import os
from tqdm import tqdm

import pickle
from scipy import interpolate

file = 'data/skyLines.txt'

wavelength = np.loadtxt(file,usecols=[0])
intensity = np.loadtxt(file,usecols=[1])
mask = (wavelength > 908) & (wavelength < 1099)
wavelengths = np.round(wavelength[mask],1)

interpolators = {}

# y incident angle
th_min=14.3*u.deg 
th_max=23.72*u.deg
th_rad = np.linspace(th_min.to(u.rad).value, th_max.to(u.rad).value, 100)
# x wavelength
MAP = np.arange(100+(14.02-10)/2,100+(22.01-10)/2,0.01)
x = MAP
y = th_rad
for i in tqdm(range(len(wavelengths))):
    z = []
    wlen = wavelengths[i]
    material = ('Vacuum', 'MgF2','Ta2O5', 'Si_Temp', 'SiO2', 'Si_Temp', 'SiO2','Epoxy','Si_Temp','Si3N4')
    n_list = np.array([index_of_refraction[m](wlen) for m in material])
    pol = 'p'
    for j in (range(len(th_rad))):
        theta = th_rad[j]
        a = []
        for i in (range(len(MAP))):

            thickness_um = np.array([np.inf, 0.1221,0.0441,MAP[i], 0.1, 0.3,1.,14,165, np.inf])
            Res = TMMSIM.coh_tmm(pol, n_list, thickness_um, theta, 1e-3 * wlen)
            Abor_prob = TMMSIM.absorp_in_each_layer(Res)[3]
            a.append(Abor_prob)
        z.append(a)

    interpolators[str(wlen)] = interpolate.interp2d(x, y, z, kind='cubic')
    
filename = 'data/Interpolator/Mono_interpolator.pkl'
outfile = open(filename,'wb')
pickle.dump(interpolators,outfile)
outfile.close()