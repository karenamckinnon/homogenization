# # Take fitted changepoints and add back into ERA5
import xarray as xr

procdir = '/home/data/projects/homogenization/proc'
target_name = 'ERA5'

land_cut = 0.5
years = 1980, 2025
nyrs = years[1] - years[0] + 1
lat_range = slice(-60, 80)

max_cps = 5
minseglen = 5 * 12  # 5 years
add_bias = True

# Land mask at 1x1
lsmask = xr.open_dataarray('/home/data/ERA5/fx/era5_lsmask.nc')
lsmask = lsmask.rename({'latitude': 'lat', 'longitude': 'lon'})
lsmask = lsmask.sortby('lat')
lsmask = lsmask.squeeze()

# Load ERA5
da_target = xr.open_dataarray('%s/ERA5_q_1x1_%i-%i.nc' % (procdir, years[0], years[1])).load()

da_target = da_target.where(lsmask > land_cut)
da_target = da_target.sel(lat=lat_range)

# Load changepoint information - magnitude
cp_file = '%s/changepoints_1x1/da_delta_at_cp_withAR_ERA5_Q2m_minus_CESM2_QREFHT.nc' % procdir
da_cp_mag = xr.open_dataarray(cp_file)

# Load changepoint information - time
cp_info_file = '%s/changepoints_1x1/ds_cp_info_ERA5_Q2m_minus_CESM2_QREFHT.nc' % procdir
ds_cp_info = xr.open_dataset(cp_info_file)

da_target_shifted = da_target.copy()
for this_cp in range(5):

    cp_time = ds_cp_info['cp_time'].sel(cp=this_cp)
    cp_mag = da_cp_mag.sel(cp=this_cp)
    cp_mag = cp_mag.fillna(0)

    if add_bias:
        cp_mag = xr.where(
            cp_mag < 0, cp_mag + 0.06,
            xr.where(cp_mag > 0, cp_mag + 0.02, cp_mag)
        )

    # Identify times post the CP
    mask = (da_target['time'] >= cp_time).astype(bool)
    increment = xr.where(mask, cp_mag, 0)

    da_target_shifted = da_target_shifted - increment

bias_str = '_addbias' if add_bias else ''

da_target_shifted.to_netcdf(
    f'{procdir}/corrected_ERA5_q_withAR{bias_str}_{years[0]}-{years[1]}.nc'
)
