import scipy.interpolate
import astropy.table
import astropy.units as u
import csv
import pandas as pd
import tmm
import numpy as np
from astropy.io import fits
import pickle
import batoid
import matplotlib.pyplot as plt

def load_refraction_data(Temp,Epoxy_ind = 1.5,kind='linear'):
    '''
    Interpolate the published refractive index of  different material
    implemented in the fringing model.

    Paremeters
    ----------
    Temp (number): Since the optical property of Silicon is temperature
        dependent, a temperature need to be specified.

    Epoxy_ind (number): The refractive index of Epoxy. Since no published data is
        available, the deflaut value is set to 1.5
    kind: The interpolating method used. The deflaut is set to 'linear'
        see scipy.interpolate.interp1d documentation for details

    Returns
    -------
    A dict that stores the interpolator for each material
    '''


    interpolators = {}
    # Vacuum.
    interpolators['Vacuum'] = lambda wlen: np.ones_like(wlen)


    ### Silicon (Temperature independent)
    abs_table = astropy.table.Table.read('data/Si-absorption.csv', format='ascii.csv', names=('wlen', 'k', 'a'))
    table = astropy.table.Table.read('data/Si-index.csv', format='ascii.csv', names=('wlen', 'n'))
    wlen = abs_table['wlen']
    k = abs_table['k']

    # Interpolate n onto the finer grid used for k.
    n = np.interp(wlen, table['wlen'], table['n'])
    # Build a linear interpolator of the complex index of refraction vs wavelength [nm].
    n = n + 1.j * k
    interpolators['Si'] = scipy.interpolate.interp1d(wlen, n, copy=True, kind=kind)


    ### Silicon (Temperature dependent)
    Green_table = astropy.table.Table.read('data/Si_index_Green.csv',format = 'ascii.csv')
    wlen_Green = Green_table['wlen']*1e3
    c_n = Green_table['C_n']
    c_k = Green_table['C_k']

    def Temp_Si_index (index, tem_coeff, T = 173):
        T0 = 300.0
        b = tem_coeff * 1e-4 * T0
        index_temp = index * (T / T0)**b
        return(index_temp)

    n_temp = Temp_Si_index(Green_table['n'],c_n,T = Temp)
    k_temp = Temp_Si_index(np.imag(interpolators['Si'](wlen_Green)),c_k,T = Temp)
    # Interpolate n onto the finer grid used for k.

    n_temp = np.interp(wlen_Green, Green_table['wlen']*1e3, n_temp)

    # Build a linear interpolator of the complex index of refraction vs wavelength [nm].
    n_temp = n_temp + 1.j * k_temp
    interpolators['Si_Temp'] = scipy.interpolate.interp1d(wlen_Green, n_temp, copy=True, kind= kind)
    ###################################

    abs_table = astropy.table.Table.read('data/Si_index_Green.csv',format = 'ascii.csv')
    wlen = abs_table['wlen']*1e3
    k = abs_table['k']
    c_n = abs_table['C_n']
    c_k = abs_table['C_k']
    def Temp_Si_index (index, tem_coeff, T = Temp):

        T0 = 300.0
        b = tem_coeff * 1e-4 * T0
        index_temp = index * (T / T0)**b

        return(index_temp)
    n_173 = Temp_Si_index(abs_table['n'],c_n,T = Temp)
    k_173 = Temp_Si_index(abs_table['k'],c_k,T = Temp)
    # Interpolate n onto the finer grid used for k.
    n_173 = np.interp(wlen, abs_table['wlen']*1e3, n_173)
    # Build a linear interpolator of the complex index of refraction vs wavelength [nm].
    n_173 = n_173 + 1.j * k_173
    interpolators['Si_Green'] = scipy.interpolate.interp1d(wlen, n_173, copy=True, kind= kind)
    ###################################

    # Read tabulated Si3N4 data from
    # http://refractiveindex.info/?shelf=main&book=Si3N4&page=Philipp
    table = astropy.table.Table.read('data/Si3N4-index.csv', format='ascii.csv', names=('wlen', 'n'))
    # Convert from um to nm.
    wlen = 1e3 * table['wlen']
    n = table['n']
    interpolators['Si3N4'] = scipy.interpolate.interp1d(wlen, n, copy=True, kind=kind)

    # Read SiO2 tabulated data from
    # http://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson
    table = astropy.table.Table.read('data/SiO2-index.csv', format='ascii.csv', names=('wlen', 'n'))
    # Convert from um to nm.
    wlen = 1e3 * table['wlen']
    n = table['n']
    interpolators['SiO2'] = scipy.interpolate.interp1d(wlen, n, copy=True, kind=kind)

    # Read MgF2 tabulated data from
    # http://refractiveindex.info/?shelf=main&book=MgF2&page=Li-o
    table = astropy.table.Table.read('data/MgF2-index.csv', format='ascii.csv', names=('wlen', 'n'))
    # Convert from um to nm.
    wlen = 1e3 * table['wlen']
    n = table['n']
    interpolators['MgF2'] = scipy.interpolate.interp1d(wlen, n, copy=True, kind=kind)

    #Epoxy
    n = np.full_like(n,1)*Epoxy_ind
    interpolators['Epoxy'] = scipy.interpolate.interp1d(wlen, n, copy=True, kind=kind)


    # Read Ta2O5 tabulated data from
    # https://refractiveindex.info/?shelf=main&book=Ta2O5&page=Rodriguez-de_Marcos
    table = astropy.table.Table.read('data/Ta2O5-index.csv', format='ascii.csv', names=('wlen', 'n','k'))
    # Convert from um to nm.
    wlen = 1e3 * table['wlen']
    n = table['n']
    k = table['k']
    n = n + 1.j*k
    interpolators['Ta2O5'] = scipy.interpolate.interp1d(wlen, n, copy=True, kind=kind)

    return interpolators


def load_ccd_map (sensor_name):
    hh = fits.open('data/sensor/%s.fits'%sensor_name)
    MAP = hh[1].data['sim']
    MAP = np.array(MAP,dtype = float)
    Fitting = MAP.reshape(3974,4000)

    unqiue_idx, return_idx = np.unique(MAP,return_inverse=True)
    return(Fitting,unqiue_idx,return_idx)

def load_interp (experiment = 'LSST'):
    if experiment == 'LSST':
        dbfile = open('data/Interpolator/LSST_interpolator.pkl', 'rb')
        interpolator = pickle.load(dbfile)
        dbfile.close()

    elif experiment == 'Mono':
        dbfile = open('data/Interpolator/Mono_interpolator.pkl', 'rb')
        interpolator = pickle.load(dbfile)
        dbfile.close()

    elif experiment == 'sky':
        dbfile = open('data/Interpolator/sky_interpolator.pkl', 'rb')
        interpolator = pickle.load(dbfile)
        dbfile.close()
    elif experiment == 'sky_mono':
        dbfile = open('data/Interpolator/sky_mono_interpolator.pkl', 'rb')
        interpolator = pickle.load(dbfile)
        dbfile.close()
        
    elif experiment == 'HSC':
        dbfile = open('data/Interpolator/HSC_200si.pkl', 'rb')  
        HSC_interpolator_200si= pickle.load(dbfile)
        dbfile.close()
        
    else:
        dbfile =  open('data/Interpolator/Normal_interpolator.pkl', 'rb')
        interpolator = pickle.load(dbfile)
        dbfile.close()
    return(interpolator)



def get_angle(theta_x,theta_y,Tele = 'LSST',plot = False):
    '''
    Specify a location on the LSST focal plane, returns the range of
    incident angle from Batoid
    '''
    if Tele == 'LSST':
        telescope = batoid.Optic.fromYaml("LSST_y.yaml")
    elif Tele == 'HSC':
        telescope = batoid.Optic.fromYaml("HSC.yaml")

    thx = np.deg2rad(theta_x)
    thy = np.deg2rad(theta_y)
    wavelength = 969.21e-9 # meters
    rays = batoid.RayVector.asPolar(
        optic=telescope,
        wavelength=wavelength,
        theta_x=thx, theta_y=thy,
        nrad=1000, naz=300  #  These control how many parallel rays are created
    )

    # Now trace through the system
    telescope.trace(rays)
    # Limit to unvignetted rays
    rays = rays[~rays.vignetted]

    dxdz_batoid = rays.vx/rays.vz
    dydz_batoid = rays.vy/rays.vz
    #plt.hist(np.arctan(np.sqrt(dxdz**2+dydz**2))*180/np.pi,label = 'Fringing')

    # We can convert these to a 1d histogram of incident angles
    inc_thx, inc_thy = batoid.utils.dirCosToField(rays.vx, rays.vy, -rays.vz)
    a,b,c= plt.hist(np.rad2deg(np.hypot(inc_thx, inc_thy)),label = 'Batoid',
        alpha = 0.3,bins = 50,density = True)
    if plot == False:
        plt.close();


    weight = []
    for i in range(len(b)-1):
        weight.append(a[i]*(b[i+1]-b[i]))

    weight = np.array(weight)
    angles = b[:-1]*np.pi/180
    return(angles,weight)
