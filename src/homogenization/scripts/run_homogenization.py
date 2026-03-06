import xarray as xr
import os
import pandas as pd
import numpy as np
from homogenization import utils as my_utils
from helpful_utilities.ncutils import lon_to_360


procdir = '/home/data/projects/homogenization/proc'
dataname = 'ERA5'  # 'ERA5'  # or HadISDH


land_cut = 0.5
years = 1980, 2024
nyrs = years[1] - years[0] + 1
lat_range = slice(-60, 80)

max_cps = 5
minseglen = 5 * 12  # 5 years

# Land mask
if dataname == 'ERA5':
    lsmask = xr.open_dataarray('/home/data/ERA5/fx/era5_lsmask_1x1.nc')

    # Load ERA5 - preprocessed in preprocess_steps
    da_target = xr.open_dataarray('%s/ERA5_q_1x1.nc' % procdir).load()
    f_nudge = 'nudged_QREFHT_rg.nc'

elif dataname == 'HadISDH':
    lsmask = xr.open_dataarray('/home/data/ERA5/fx/era5_lsmask_5x5_conservative.nc')
    da_target = xr.open_dataset('/home/data/HadISDH/landq/HadISDH.landq.4.6.1.2024f_FLATgridHOM5by5_anoms9120.nc')
    da_target = da_target['q_abs']
    da_target = da_target.rename({'latitude': 'lat', 'longitude': 'lon'})
    da_target = lon_to_360(da_target)
    da_target = da_target.sel(time=slice('%i' % years[0], '%i' % years[1])).load()
    f_nudge = 'nudged_QREFHT_rg_to_HadISDH.nc'

lsmask = lsmask.squeeze()

# Load nudged dataset; Data are regridded in preprocess_steps
da_nudge = xr.open_dataarray('/home/data/SIMULATIONS/Nudged_CESM2/mon/%s' % f_nudge)
# fix timestamp issue, move to regular calendar
new_time = pd.date_range(start='1979-01-01',
                         periods=da_nudge.sizes['time'],
                         freq='MS')

if dataname == 'HadISDH':
    new_time += np.timedelta64(15, 'D')  # Hadley timestamps in middle of month

da_nudge = 1000 * da_nudge.assign_coords(time=new_time)  # g/kg
da_nudge = da_nudge.sel(time=slice('%i' % years[0], '%i' % years[1])).load()

delta = da_target - da_nudge
delta = delta.where(lsmask > land_cut)
delta = delta.sel(lat=lat_range)

if dataname == 'HadISDH':  # mask places that don't have complete coverage
    missing_count = np.isnan(delta).sum('time')
    delta = delta.where(missing_count == 0)

delta_norm = my_utils.normalize_delta(delta)
delta_mu_norm = my_utils.normalize_delta(delta, do_std=False)  # for fitting model

# Load just in case
delta_norm = delta_norm.load()

# # Run algorithm on masked deltas
savename = '%s/ds_cp_info_%s_%i-%i.nc' % (procdir, dataname, years[0], years[1])
if os.path.isfile(savename):
    ds_cp_info = xr.open_dataset(savename)
else:
    # R function to use
    r_fn_envcpt = my_utils.get_R_fn_envcpt_ar1()
    ds_cp_info = my_utils.return_changepoint_info(da=delta_norm,
                                                  fn=r_fn_envcpt,
                                                  max_cps=max_cps,
                                                  minseglen=minseglen)
    ds_cp_info.to_netcdf(savename)

# # Fit correct model to unnormalized data to identify changepoint delta
da_delta_at_cp, fitted_da = my_utils.get_delta_mag(ds_cp_info, delta_mu_norm)

da_delta_at_cp.to_netcdf('%s/da_delta_at_cp_withAR_%s_%i-%i.nc' % (procdir, dataname, years[0], years[1]))
fitted_da.to_netcdf('%s/fitted_da_withAR_%s_%i-%i.nc' % (procdir, dataname, years[0], years[1]))
