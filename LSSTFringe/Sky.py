import numpy as np
import galsim
from lsst.sims.photUtils import BandpassDict, Sed
import lsst.sims.skybrightness as skybrightness
import lsst.sims.photUtils.PhotometricParameters as phopara
import lsst.sims.utils.ObservationMetaData as ObsMeta
import lsst.sims.utils as utils


class Sky():
    """
    SkyModel Class

    Parameters
    ----------
    """

    def __init__(self,
        ra = 0.2, dec =0.2, Mjd = 60000.15,
        compList = ['scatteredStar','lowerAtm','upperAtm','airglow','mergedSpec'
        ,'twilight','zodiacal','moon']):

        self.ra = ra
        self.dec = dec
        self.Mjd = Mjd
        self.compList = compList

    def Phot_par(self):
        '''
        Retrun
        ------
        Photometric Parameters
        '''
        return(phopara(bandpass='y'))

    def detector_throughputs(self):
        '''
        Return
        ------
        The throughput curve for detector Quamtum efficiency (QE) and
        Anti-Reflection coating (AR)
        '''
        detector_f = 'data/throughputs/baseline/detector.dat'
        throughputs = np.loadtxt(detector_f,usecols=[1])
        return(throughputs)

    def Bandpass(self):
        bandpassdict = BandpassDict.loadBandpassesFromFiles()[0]
        # We want the yband
        bandpass = bandpassdict['y']

        return(bandpass)

    def sky_counts_per_sec(self,skyModel, photParams, bandpass, magNorm=None):
        """
        This is the code borrowed from LSST imsim skyModel.py.
        Original code see:
        https://github.com/LSSTDESC/imSim/blob/master/python/desc/imsim/skyModel.py

        Compute the sky background counts per pixel per second.  Note that
        the gain in photParams is applied to the return value such that
        "counts" are in units of ADU.
        Parameters
        ----------
        skyModel: lsst.sims.skybrightness.SkyModel
            Model of the sky for the current epoch.
        photParams: lsst.sims.photUtils.PhotometricParameters
            Object containing parameters of the photometric response of the
            telescope, including pixel scale, gain, effective area, exposure
            time, number of exposures, etc.
        bandpass: lsst.sims.photUtils.Bandpass
            Instrumental throughput for a particular passband.
        magNorm: float [None]
            If not None, then renormalize the sky SED to have a monochromatic
            magnitude of magNorm at 500nm.  Otherwise, use the default
            skyModel normalization.
        Returns
        -------
        ADUs per second per pixel
        """
        wave, spec = skyModel.returnWaveSpec()
        sed = Sed(wavelen=wave, flambda=spec[0, :])
        if magNorm is not None:
            flux_norm = sed.calcFluxNorm(magNorm, bandpass)
            sed.multiplyFluxNorm(flux_norm)
        countrate_per_arcsec = sed.calcADU(bandpass=bandpass, photParams=photParams)
        exptime = photParams.nexp*photParams.exptime
        return countrate_per_arcsec*photParams.platescale**2/exptime
    
    def get_flux(self,skyModel, bandpass):

        wave, spec = skyModel.returnWaveSpec()
        sed = Sed(wavelen=wave, flambda=spec[0, :])

        flux = sed.calcFlux(bandpass=bandpass)
    
        return flux

    def set_skyspec(self):
        sm = skybrightness.SkyModel(lowerAtm=True, upperAtm=True,airglow=True,
            scatteredStar=True,mergedSpec=False)

        # Set Ra Dec and Mjd, can change later
        sm.setRaDecMjd(self.ra,self.dec,self.Mjd)

        return(sm)

    def Count_sky_continuum(self):
        sm = self.set_skyspec()
        for comp in self.compList: setattr(sm, comp, True)
        # Turn off upperAtm
        sm.mergedSpec= False
        sm.upperAtm = False
        sm.airglow = False
        # Compute the spectra
        sm._interpSky()
        CountSkyBg = self.sky_counts_per_sec(skyModel=sm,
            photParams=self.Phot_par(),bandpass=self.Bandpass())

        return(CountSkyBg)

    def Count_upper_atm(self):
        sm = self.set_skyspec()
        # Upper Atm spectrum
        for comp in self.compList: setattr(sm, comp, False)
        sm.mergedSpec= False

        # Turn on upper atm
        sm.upperAtm = True
        sm.airglow = True
        # Compute the spectra
        sm._interpSky()

        bandpass=self.Bandpass()
        # Count the detector QE and AR coating
        detector = self.detector_throughputs()
        bandpass.sb = bandpass.sb/detector
        where_are_NaNs = np.isnan(bandpass.sb)
        bandpass.sb[where_are_NaNs] = 0
        bandpass.phi = None

        CountUpperAtm = self.sky_counts_per_sec(skyModel=sm,
            photParams= self.Phot_par(),bandpass=bandpass)


        return(CountUpperAtm,sm.wave,sm.spec)
    
    def flux_upper_atm(self):
        sm = self.set_skyspec()
        # Upper Atm spectrum
        for comp in self.compList: setattr(sm, comp, False)
        sm.mergedSpec= False

        # Turn on upper atm
        sm.upperAtm = True
        sm.airglow = True
        # Compute the spectra
        sm._interpSky()

        bandpass=self.Bandpass()
        # Count the detector QE and AR coating
        detector = self.detector_throughputs()
        bandpass.sb = bandpass.sb/detector
        where_are_NaNs = np.isnan(bandpass.sb)
        bandpass.sb[where_are_NaNs] = 0
        bandpass.phi = None
        FLUX = self.get_flux(skyModel=sm,bandpass = bandpass)
        return(FLUX)

class OHlines():
    """
    OH lines
    """

    def __init__(self,
        vib_group = ['7-3','8-4','3-0','9-5','4-1','5-2','Unidentified']):
        self.vib_group = vib_group

    def get_skyline_file(self):
        """
        Load skyline from PFS with line identification.
        """
        file = 'data/ybandlines.txt'

        return(file)

    def load_skyline(self):
        '''
        Load line intensity and wavelength from skyline file
        '''
        file = self.get_skyline_file()
        wavelength = np.loadtxt(file,usecols=[0])
        intensity = np.loadtxt(file,usecols=[1])
        group_id = np.loadtxt(file,usecols=[4])

        return(wavelength,intensity,group_id)

    def get_grouping(self,group):
        '''
        Get the line grouping mask
        '''
        group_dir = {}
        group_dir['7-3'] = (group == 73)
        group_dir['8-4'] = (group == 84)
        group_dir['3-0'] = (group == 30)
        group_dir['9-5'] = (group == 95)
        group_dir['4-1'] = (group == 41)
        group_dir['5-2'] = (group == 52)
        group_dir['Unidentified'] = (group == 0)

        group73 = (group == 73)
        group84 = (group == 84)
        group30 = (group == 30)
        group95 = (group == 95)
        group41 = (group == 41)
        group52 = (group == 52)
        group_n = (group == 0)

        groups = [group73,group84,group30,group95,group41,group52,group_n]

        return(groups,group_dir)

    def get_line(self):
        '''
        Get OH lines in each group
        '''
        lines_int = {}
        lines_wlen = {}
        wavelen,intensity,group = self.load_skyline()
        group_mask,group_dir = self.get_grouping(group)

        for g,l in zip(group_mask, self.vib_group):
            lines_int[l] = intensity[g]
            lines_wlen[l] = wavelen[g]

        return(lines_wlen,lines_int,group_dir)

class Conv():
    """
    Apply band throughput to line intensity
    """

    def __init__(self, band_wlen,band_thr,line_wlen,line_int):
        self.band_wlen = band_wlen
        self.band_thr = band_thr
        self.line_int = line_int
        self.line_wlen = line_wlen

    def conv(self,verbose = False):
        # Restrict wavelength range to LSST yband
        wlen_mask = (self.line_wlen > 908) & (self.line_wlen < 1099)
        # Round to first decimal to avoid numerical issue
        band_round = np.round(self.band_wlen,1)
        yband_range = np.round(self.line_wlen[wlen_mask],1)

        # Calculate sky line intensity under bandpass
        spec = []

        for i in range(len(yband_range)):
            indx = np.where(band_round == yband_range[i])[0]

            if verbose == True:
                print(self.band_thr[indx],self.line_int[wlen_mask][i],
                self.band_thr[indx]*self.line_int[wlen_mask][i])

            val = self.band_thr[indx]*self.line_int[wlen_mask][i]
            spec.append(val[0])
        spec = np.array(spec)

        return(spec)
