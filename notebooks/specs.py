# from spectrometers.ipynb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xarray as xr
import holoviews as hv
hv.extension("bokeh", logo=False)
import matplotlib.pyplot as plt

from datetime import datetime, timezone
from scipy.interpolate import interp1d

from fastcore.foundation import patch
from fastcore.dispatch import typedispatch # only types of first two arguments used
from tqdm import tqdm

from holoviews import opts
from holoviews.streams import Pipe, Buffer

from typing import Iterable, Union, Callable, List, TypeVar, Generic, Tuple, Optional


class DateTimeBuffer():
    """Records timestamps in UTC time."""
    def __init__(self, n:int = 16) -> None:
        """Initialise a nx1 array and write index"""
        self.data = np.arange(n).astype(datetime)
        self.n = n
        self.write_pos = 0
        
    def __getitem__(self, key:slice) -> datetime:
        return self.data[key]

    def update(self) -> None:
        """Stores current UTC time in an internal buffer when this method is called."""
        ts = datetime.timestamp(datetime.now())
        self.data[self.write_pos] = datetime.fromtimestamp(ts, tz=timezone.utc)
        self.write_pos += 1

        # Loop back if buffer is full
        if self.write_pos == self.n:
            self.write_pos = 0
            


class SpectraBuffer(object):
    """Circular FIFO buffer for spectral measurements"""
    def __init__(self, 
                 nbands:int=8, # number of spectral pixels
                 nlines:int=4, # length of buffer
                 dtype:type = np.uint16 # data type. raw counts use unit16 but radiance use float32
                ) -> None:
        """Preallocate data array"""
        self.size = (nlines,nbands)
        self.data = np.zeros(self.size,dtype=dtype)
        self.write_pos = 0
        self.read_pos = 0
        self.slots_left = nlines
        self.time_buff = DateTimeBuffer(n=nlines)
        
    def __getitem__(self, key:slice):
        return self.data[key]
    
    def _inc(self, idx:int) -> int:
        """Increment read/write index with wrap around"""
        idx += 1
        if idx == self.size[0]: idx = 0
        return idx
    
    def is_empty(self) -> bool:
        return self.slots_left == self.size[0]

    def put(self, line:np.ndarray) -> None:
        """Place spectra into the buffer"""
        self.data[self.write_pos,:] = line
        self.time_buff.update()
        
        # if buffer full, update read position to keep track of oldest slot
        self.slots_left -= 1
        if self.slots_left < 0:
            self.slots_left = 0
            self.read_pos = self._inc(self.read_pos)
        
        self.write_pos = self._inc(self.write_pos)
            
    def get(self) -> np.ndarray:
        """Reads the oldest (n-1)darray from the buffer"""
        if self.slots_left < self.size[0]:
            val = self.data[self.read_pos,:]
            self.slots_left += 1
            self.read_pos = self._inc(self.read_pos)
            return val
        else:
            return None
        

class OceanSpectro(SpectraBuffer):
    
    def __init__(self, 
                 exposure:int=40, # exposure time in ms
                 nlines:int=128,  # length of buffer to preallocate
                 serial_num:str="FLMS01766", # spectrometer name "FLMS01766" or "QEP00994"
                 trigger:bool=False,         # trigger on rising edge or not (normal)
                 radiometric:bool=False,     # radiance stored using floating point or raw counts using integers
                ) -> None:
        """
        Initialise the Ocean Insight spectrometers. 
        """
        from seabreeze.spectrometers import Spectrometer
        try: 
            self.spec = Spectrometer.from_serial_number(serial_num) if serial_num else Spectrometer.from_first_available()
            self.spec.integration_time_micros(exposure*1000)
            self._exposure = exposure
            self.trigger(trigger)

            self.wavelengths = self.spec.wavelengths()
            self.get_spectra()
        except: 
            print("Device already opened. Close with `self.close()` then try again.")
            
        self.n = nlines
        super().__init__(nbands=len(self.wavelengths),nlines=nlines,dtype=np.float32 if radiometric else np.int32)
        
        
    @property
    def exposure(cls):
        return cls._exposure
    @exposure.setter
    def exposure(cls,val):
        cls._exposure = val
        cls.spec.integration_time_micros(val*1000)

    def get_spectra(self) -> np.array:
        """Grab spectral measurment"""
        self.start()
        self.last_spectra = self.spec.intensities()
        self.close()
        return self.last_spectra
    
    def start(self):
        """open spectrometer if closed"""
        try: self.spec.open()
        except: print("Please initialise the spectrometer first")
        
    def close(self):
        """close connection to spectrometer"""
        self.spec.close()
        
    def __repr__(self) -> str:
        return f"Spectrometer {self.spec.serial_number}. Exposure time = {self.exposure} ms. Wavelength range = {self.wavelengths[0]:.2f} to {self.wavelengths[-1]:.2f} nm."
    
    def trigger(self, 
                val:bool=False, # trigger mode
               ) -> None:
        """normal mode or hardware rising edge trigger mode"""
        # see https://www.oceaninsight.com/globalassets/catalog-blocks-and-images/manuals--instruction-ocean-optics/electronic-accessories/external-triggering-options_firmware3.0andabove.pdf
        if val:
            self.spec.trigger_mode(3)
        else:
            self.spec.trigger_mode(0)
            
    def collect(self) -> None:
        """Fill up the spectral buffer"""
        self.start()
        for i in tqdm(range(self.n)):
            self.put( self.spec.intensities() )
        self.close()
        
    def average(self,n:int) -> np.array:
        """Measure `n` spectra and return the average."""
        temp = np.zeros((n,len(self.wavelengths)),dtype=np.float32)
        self.start()
        for i in tqdm(range(n)):
            temp[i,:] = self.spec.intensities()
        self.close()
        return np.mean(temp,axis=0)
        
    def show(self,
             plot_lib="bokeh", # plotting backend 'bokeh' or 'matplotlib'
             savedir:str=None, # save directory string
            ) -> hv.Curve:     # plot object
        """Plot spectra with option to save plot"""
        hv.extension(plot_lib, logo=False)
        curve = hv.Curve( (self.wavelengths,self.last_spectra) ).opts(xlabel="wavelength (nm)",ylabel="counts")
        
        if plot_lib == "bokeh":
            curve = curve.opts(width=1000,height=250)
        else: # plot_lib == "matplotlib"
            curve = curve.opts(fig_inches=12,aspect=3)
        if savedir:
            curve.savefig(f"{savedir}/{self.time_buff.data[0].strftime('%Y_%m_%d')}/{self.spec.serial_number}_{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.pdf",bbox_inches='tight', pad_inches=0)
        return curve
    
    def waterfall(self,
                  plot_lib:str="bokeh", # plotting backend 'bokeh' or 'matplotlib'
                  wavelen_range:tuple=None, # wavelength nm range as tuple (start_nm, end_nm)
                  savedir:str=None, # save directory string. only possible with plotlib='matplotlib'
                ) -> hv.Image:      # plot object
        """Plot spectral timeseries"""
        hv.extension(plot_lib, logo=False)
        def rescale(x, in_min, in_max, out_min, out_max):
            return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
        if not wavelen_range:
            start_idx = 0
            end_idx   = len(self.wavelengths)
            start_wavelen, end_wavelen = self.wavelengths[0], self.wavelengths[-1]
        else:
            start_idx = np.sum(self.wavelengths<wavelen_range[0])
            end_idx   = np.sum(self.wavelengths<wavelen_range[1])
            start_wavelen, end_wavelen = wavelen_range
        img = hv.Image( self.data[:,start_idx:end_idx] ).opts(colorbar=True,cmap="Viridis",
            xticks=[(i,int(rescale(i,-0.5,0.5,start_wavelen,end_wavelen))) for i in np.arange(-0.5,0.51,0.08)],
             yticks=[(i,np.round(rescale(i,-0.5,0.5,0,self.n*(self.exposure+32)/1000),1)) for i in np.arange(-0.5,0.51,0.1)] )
        
        if plot_lib == "bokeh":
            img = img.opts(width=500,height=500)
        else: # plot_lib == "matplotlib"
            img = img.opts(fig_inches=12,aspect=1.)
        if savedir:
            fig, ax = plt.subplots(figsize=(10,10))
            ax.imshow(self.data[:,start_idx:end_idx]); ax.set_xlabel("wavelength index"); ax.set_ylabel("time index")
            fig.savefig(f"{savedir}/{self.spec.serial_number}_{self.time_buff.data[0].strftime('%Y_%m_%d-%H_%M_%S')}.png",bbox_inches='tight', pad_inches=0)
        return img
    
    def get_live_waterfall(self) -> hv.DynamicMap:
        """Produce a dynamic live image that is updated when `self.runtime` is called."""
        def rescale(x, in_min, in_max, out_min, out_max):
            return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
        self.spec_stream = Pipe([])
        self.img_dmap = hv.DynamicMap(hv.Image,streams=[self.spec_stream],).opts(xlabel="wavelength (nm)",ylabel="seconds ago")
        self.img_dmap.opts(xlim=(-0.5,0.5),ylim=(-0.5,0.5),width=500,height=500,cmap="plasma",colorbar=True,toolbar='above',
             xticks=[(i,int(rescale(i,-0.5,0.5,self.wavelengths[0],self.wavelengths[-1]))) for i in np.arange(-0.5,0.51,0.08)],
             yticks=[(i,np.round(rescale(i,-0.5,0.5,0,self.n*self.exposure/1000),1)) for i in np.arange(-0.5,0.51,0.1)], yformatter='%.1f s',
             title=f"Spectrometer {self.spec.serial_number}. Exposure time = {self.exposure} ms.")
        return self.img_dmap
    
    def runtime(self,
                seconds:float=10, # seconds to run the spectrometer
               ) -> None:
        """Run the spectrometer for `seconds` long and fill up the spectral buffer for `self.img_dmap`."""
        self.start()
        for i in tqdm(range(int(seconds*1000/self.exposure))):
            self.put( self.spec.intensities() )
            temp = self.data.copy()
            temp[:self.n-self.read_pos,:] = self.data[self.read_pos:,:]
            temp[self.n-self.read_pos:,:] = self.data[:self.read_pos,:]
            self.spec_stream.send(temp)
        self.close()

@patch
def get_AC_diff(self:OceanSpectro) -> None:
    self.diff_arr = np.abs( np.float32(self.data[1:,:])-np.float32(self.data[:-1,:]) )
    return self.diff_arr

@patch
def save(self:OceanSpectro,
        savedir:str = ".",
        suffix:str = "") -> None:
    """save spectral buffer into HDF5 format"""
    attrs = {"exposure":self.exposure,"spectrometer":self.spec.serial_number}
    #self.directory = self.directory = f"{savedir}/{self.time_buff.data[0].strftime('%Y_%m_%d')}"
    
    self.coords = dict(wavelength=(["wavelength"],self.wavelengths),
                       time=(["time"],self.time_buff.data.astype(np.datetime64)) )
    self.nc = xr.Dataset(data_vars=dict(datacube=(["time","wavelength"],self.data)),
                             coords=self.coords, 
                             attrs=attrs)
    self.nc.time.attrs["long_name"]   = "UTC time"
    self.nc.time.attrs["description"] = "UTC time for each measurement"
    self.nc.wavelength.attrs["long_name"]   = "wavelength_nm"
    self.nc.wavelength.attrs["units"]       = "nanometers"
    self.nc.wavelength.attrs["description"] = "wavelength in nanometers."
    
    self.nc.to_netcdf(f"{savedir}/{self.spec.serial_number}_{self.time_buff.data[0].strftime('%Y_%m_%d-%H_%M_%S')}_spectra{suffix}.h5")
    self.waterfall(plot_lib="matplotlib",savedir=savedir)
    
@patch
def load(self:OceanSpectro,
        fname:str, # path to h5 or netcdf4 file
        ) -> None:
    ds = xr.open_dataset(fname)
    self._exposure = ds.attrs["exposure"]
    self.wavelengths = ds.coords["wavelength"]
    self.n = ds.datacube.shape[0]
    self.size = ds.datacube.shape
    self.data = ds.datacube.data
    self.write_pos = ds.datacube.shape[1]-1
    self.read_pos  = 0
    self.slots_left = self.n
    self.time_buff.data = ds.coords["time"]
    
@patch
def measure_dark(self:OceanSpectro,
                fname:str=None) -> None:
    x = self.average(100)
    self.dark_df = pd.DataFrame(x,columns=["dark"])
    if savedir:
        self.dark_df.to_csv(fname)
    return self.dark_df


@patch
def load_dark(self:OceanSpectro,
              fname:str) -> None:
    self.dark_df = pd.read_csv(fname)
    
    
@patch
def measure_sphere(self:OceanSpectro,
                   fname:str=None) -> None:
    x = self.average(100)
    self.sphere_df = pd.DataFrame(x,columns=["sphere"])
    if savedir:
        self.sphere_df.to_csv(fname)
    return self.sphere_df


@patch
def load_sphere(self:OceanSpectro,
                fname:str) -> None:
    self.sphere_df = pd.read_csv(fname)
    
    
    

    
