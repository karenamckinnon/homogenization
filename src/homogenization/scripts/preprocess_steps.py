"""Preprocess ERA5 and CESM2 nudged sim"""

import xesmf as xe
import xarray as xr
from glob import glob
import os
import numpy as np


def lon_to_360(da, lon_name='lon'):
    """Returns the xr.DataArray with longitude changed from -180, 180 to 0, 360"""
    da = da.assign_coords({lon_name: (da[lon_name] + 360) % 360})
    return da.sortby(lon_name)


def q_from_Td_p(Td, p, return_vp=False):
    """
    Calculate specific humidity using approximations in McKinnon et al 2021, NCC.

    Parameters
    ----------
    Td : float or numpy array or xarray datarray
        Dew point temperature in Celsius
    p : float or numpy array or xarray dataarray
        Surface pressure (hPa)
    return_vp : bool
        Return vapor pressure (hPa)?

    Returns
    -------
    q : float or numpy array or xarray datarray
        Specific humidity (g/kg)
    """

    vp = 6.112 * np.exp((17.67 * Td) / (Td + 243.5))
    q = 1000 * 0.622 * vp / (p - 0.378 * vp)

    if return_vp:
        return q, vp
    else:
        return q


years = 1980, 2025
# # Calculate ERA5 q from T2m and surface pressure
era5_dir = '/home/data/ERA5/month'

var_list = '2m_dewpoint_temperature', 'surface_pressure'
ds_era5 = []
for v in var_list:
    da = xr.open_dataarray('%s/%s/%s.nc' % (era5_dir, v, v))
    da = da.rename({'valid_time': 'time', 'latitude': 'lat', 'longitude': 'lon'})
    da = da.sortby('lat')
    da = da.sel(time=slice('%04i' % years[0], '%04i' % years[1]))
    print(da.units)
    ds_era5.append(da.rename(v))
ds_era5 = xr.merge(ds_era5)

ds_era5['q'] = q_from_Td_p(ds_era5['2m_dewpoint_temperature'] - 273.15,
                           ds_era5['surface_pressure'] / 100)


# # Regrid ERA5 to 1x1
lat_use = np.arange(-89.5, 90, 1)
lon_use = np.arange(0.5, 360, 1)
wgt_file = '/home/data/ERA5/wgts_0.25_to_1.nc'

if os.path.isfile(wgt_file):
    reuse_weights = True
else:
    reuse_weights = False

regridder = xe.Regridder({'lat': ds_era5['q'].lat, 'lon': ds_era5['q'].lon},
                         {'lat': lat_use, 'lon': lon_use},
                         'bilinear',
                         periodic=True, reuse_weights=reuse_weights,
                         filename=wgt_file)

da_rg = regridder(ds_era5['q'])
da_rg = da_rg.rename('Q')

da_rg.to_netcdf('/home/data/projects/homogenization/proc/ERA5_q_1x1_%i-%i.nc' % (years[0], years[1]))


# # Regrid CESM2 to other grids
regrid_to = 'HadISDH'  # or HadISDH
if regrid_to == 'ERA5':
    era5_dir = '/home/data/ERA5/month/q/1x1'
    files = sorted(glob('%s/*.nc' % era5_dir))
    da_era5 = xr.open_mfdataset(files)['q']

    lat_use = da_era5.lat.values
    lon_use = da_era5.lon.values
    del da_era5
elif regrid_to == 'HadISDH':
    had_file = '/home/data/HadISDH/landq/HadISDH.landq.4.6.1.2024f_FLATgridHOM5by5_anoms9120.nc'
    da_had = lon_to_360((xr.open_dataset(had_file)['q_anoms']).rename({'latitude': 'lat', 'longitude': 'lon'}))

    lat_use = da_had.lat.values
    lon_use = da_had.lon.values
    del da_had


base_dir = '/home/data/SIMULATIONS/Nudged_CESM2/mon'

wgt_file = '%s/xe_weights_from_CESM2_to_%s.nc' % (base_dir, regrid_to)

if os.path.isfile(wgt_file):
    reuse_weights = True
else:
    reuse_weights = False

files = sorted(glob('%s/f.e21.FHIST*QREFHT*' % base_dir))
print(files)
da = xr.open_mfdataset(files)['QREFHT']
regridder = xe.Regridder({'lat': da.lat, 'lon': da.lon},
                         {'lat': lat_use, 'lon': lon_use},
                         'bilinear',
                         periodic=True, reuse_weights=reuse_weights,
                         filename=wgt_file)

da = regridder(da)
da = da.rename('QREFHT')

da.to_netcdf('%s/nudged_QREFHT_rg_to_%s.nc' % (base_dir, regrid_to))
