# # For realistic levels of variability, trend, and autocorrelation, test CP detection algorithm
import os
import xarray as xr
import numpy as np
from helpful_utilities import xutils
import homogenization.utils as my_utils


# ## Parameters
procdir = '/home/data/projects/homogenization/proc'

max_cps = 5
minseglen = 5 * 12  # 5 years

synth_cp_std = 0.5
seed = 27

# ## Load actual data and calculate key characteristics: trend, seasonal cycle in variance, ar1 coefficient
f_metrics = '%s/residual_metrics.nc' % procdir
if os.path.isfile(f_metrics):
    ds_metrics = xr.open_dataset(f_metrics)
else:
    delta_f = '%s/ERA5_nudged_delta.nc' % procdir
    da_delta = xr.open_dataarray(delta_f)
    da_delta = da_delta.load()

    # Remove differences in mean seasonal cycle
    da_delta = da_delta.groupby('time.month') - da_delta.groupby('time.month').mean()

    # For each gridbox, calculate trend, variance, AR(1)
    trend, _ = xutils.xr_linregress_pval(da_delta, time_dim='time')

    # remove trend to get variance
    da_detrended = xutils.detrend_xarray(da_delta, dim='time')

    std_by_month = da_detrended.groupby('time.month').std()

    # get autocorrelation -- shift by 1
    ar1 = xr.corr(da_detrended.shift(time=1), da_detrended, dim='time')

    # Save these metrics
    ds_metrics = xr.merge((trend.rename('trend per year'),
                           std_by_month.rename('sigma'),
                           ar1.rename('AR1 coeff')))
    ds_metrics.to_netcdf('%s/residual_metrics.nc' % procdir)

# ## Create synthetic dataarray with these properties
savename = '%s/synth_residuals_seed-%i.nc' % (procdir, seed)
if os.path.isfile(savename):
    da_synthetic = xr.open_dataarray(savename)
else:
    np.random.seed(seed)
    n_time = len(da_delta['time'])
    da_synthetic = xr.apply_ufunc(
        my_utils.make_synthetic_ts,
        da_delta['time'],
        ds_metrics['trend per year'],
        ds_metrics['AR1 coeff'],
        ds_metrics['sigma'],
        input_core_dims=[['time'], [], [], ['month']],
        output_core_dims=[['time']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        output_sizes={'time': n_time},
    )

    da_synthetic = da_synthetic.assign_coords(time=da_delta['time'])
    da_synthetic.to_netcdf(savename)

da_synthetic = da_synthetic.transpose('time', 'lat', 'lon')
# ## Add in changepoints
da_synthetic_with_cps, t_idx, cp_2d = my_utils.add_synth_cp(da_synthetic, synth_cp_std, minseglen, seed=(seed + 1))

# ## Save synthetic inputs
t_idx.to_netcdf('%s/synth_cps_time_sigma-%0.1f_seed-%i.nc' % (procdir, synth_cp_std, seed))
cp_2d.to_netcdf('%s/synth_cps_cpmag_sigma-%0.1f_seed-%i.nc' % (procdir, synth_cp_std, seed))

# ## Remove climatologies in mean and variance
da_synthetic_with_cps_norm = my_utils.normalize_delta(da_synthetic_with_cps)

# ## Run CP algorith
# R function to use
r_fn_envcpt = my_utils.get_R_fn_envcpt_ar1()
ds_cp_info = my_utils.return_changepoint_info(da=da_synthetic_with_cps_norm,
                                              fn=r_fn_envcpt,
                                              max_cps=max_cps,
                                              minseglen=minseglen)

# ## Fit changepoint model to original data with only climatological mean removed
da_synthetic_with_cps_norm_mean = my_utils.normalize_delta(da_synthetic_with_cps, do_std=False)
da_delta_at_cp, fitted_da = my_utils.get_delta_mag(ds_cp_info, da_synthetic_with_cps_norm_mean)

da_delta_at_cp.to_netcdf('%s/da_delta_at_cp_synth_sigma-%0.1f_seed-%i.nc' % (procdir, synth_cp_std, seed))
ds_cp_info.to_netcdf('%s/ds_cp_info_synth_sigma-%0.1f_seed-%i.nc' % (procdir, synth_cp_std, seed))
fitted_da.to_netcdf('%s/fitted_da_synth_sigma-%0.1f_seed-%i.nc' % (procdir, synth_cp_std, seed))
