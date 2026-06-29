#!/usr/bin/env python
"""Run changepoint homogenization from the command line.

Examples
--------
Run directly on one target file:

    python run_homogenization_cli.py target.nc --output-dir proc

Run on target minus nudged data:

    python run_homogenization_cli.py target.nc \
        --nudge-file nudged.nc \
        --output-dir proc \
        --max-cps 5 \
        --minseglen 60

Use mean-centering only, rather than mean/std normalization, for changepoint
identification:

    python run_homogenization_cli.py target.nc --no-do-std
"""

from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr
import numpy as np
from homogenization import utils as my_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Run changepoint homogenization on a target DataArray. If a nudged '
            'file is supplied, the algorithm is run on target minus nudged.'
        )
    )

    parser.add_argument(
        'target_file',
        type=Path,
        help='NetCDF file containing the DataArray to homogenize.',
    )
    parser.add_argument(
        '--nudge-file',
        type=Path,
        default=None,
        help=(
            'Optional NetCDF file containing nudged data. If supplied, the '
            'changepoint algorithm is applied to target - nudged.'
        ),
    )
    parser.add_argument(
        '--target-var',
        default=None,
        help=(
            'Variable name for the target data if target_file is a Dataset. '
            'If omitted, the file must contain a DataArray or a Dataset with '
            'one data variable.'
        ),
    )
    parser.add_argument(
        '--nudge-var',
        default=None,
        help=(
            'Variable name for the nudged data if nudge_file is a Dataset. '
            'If omitted, the file must contain a DataArray or a Dataset with '
            'one data variable.'
        ),
    )
    parser.add_argument(
        '--nudge-scale',
        type=float,
        default=1.0,
        help='Factor applied to nudged data before subtraction. Default: 1.0.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('.'),
        help='Directory where output NetCDF files will be written. Default: current directory.',
    )
    parser.add_argument(
        '--output-prefix',
        default=None,
        help='Prefix for output files. Default is inferred from the input filenames and settings.',
    )
    parser.add_argument(
        '--max-cps',
        type=int,
        default=5,
        help='Maximum number of changepoints passed to return_changepoint_info(). Default: 5.',
    )
    parser.add_argument(
        '--minseglen',
        type=int,
        default=5 * 12,
        help='Minimum segment length passed to return_changepoint_info(). Default: 60.',
    )
    
    parser.add_argument(
        '--envcpt-models',
        nargs='+',
        choices=('meanar1', 'trendar1'),
        default=('meanar1', 'trendar1'),
        help=(
            'EnvCpt AR(1) model(s) to allow. '
            'Use --envcpt-models trendar1 for trend-AR(1)-only sensitivity. '
            'Default: meanar1 trendar1.'
        ),
    )
    
    norm_group = parser.add_mutually_exclusive_group()
    norm_group.add_argument(
        '--do-std',
        action='store_true',
        dest='do_std',
        default=True,
        help='Normalize by both mean and standard deviation for changepoint detection. Default.',
    )
    norm_group.add_argument(
        '--no-do-std',
        action='store_false',
        dest='do_std',
        help='Normalize without standardizing by the standard deviation, i.e. do_std=False.',
    )

    parser.add_argument(
        '--land-mask',
        type=Path,
        default=None,
        help='Optional NetCDF file containing a land mask DataArray.',
    )
    parser.add_argument(
        '--land-mask-var',
        default=None,
        help='Variable name for the land mask if land_mask is a Dataset.',
    )
    parser.add_argument(
        '--land-cut',
        type=float,
        default=0.5,
        help='Land mask threshold used as land_mask > land_cut. Default: 0.5.',
    )
    parser.add_argument(
        '--lat-range',
        type=float,
        nargs=2,
        metavar=('LAT_MIN', 'LAT_MAX'),
        default=None,
        help='Optional latitude range to retain, e.g. --lat-range -60 80.',
    )
    parser.add_argument(
        '--start-year',
        type=int,
        default=None,
        help='Optional first year to retain from the time coordinate.',
    )
    parser.add_argument(
        '--end-year',
        type=int,
        default=None,
        help='Optional final year to retain from the time coordinate.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Recompute ds_cp_info even if the output file already exists.',
    )
    parser.add_argument(
        '--no-load',
        action='store_true',
        help='Do not explicitly load input arrays into memory before running the algorithm.',
    )

    return parser.parse_args()


def open_dataarray(filename: Path, varname: str | None = None) -> xr.DataArray:
    """Open a DataArray from either a DataArray file or a single-variable Dataset."""
    if not filename.exists():
        raise FileNotFoundError(f'File not found: {filename}')

    if varname is not None:
        ds = xr.open_dataset(filename)
        if varname not in ds:
            available = ', '.join(ds.data_vars)
            raise KeyError(f'{varname!r} not found in {filename}. Available variables: {available}')
        return ds[varname]

    try:
        return xr.open_dataarray(filename)
    except ValueError:
        ds = xr.open_dataset(filename)
        data_vars = list(ds.data_vars)
        if len(data_vars) != 1:
            available = ', '.join(data_vars)
            raise ValueError(
                f'{filename} contains multiple variables. Specify one with --target-var, '
                f'--nudge-var, or --land-mask-var. Available variables: {available}'
            )
        return ds[data_vars[0]]


def subset_time(da: xr.DataArray, start_year: int | None, end_year: int | None) -> xr.DataArray:
    """Subset a DataArray by year strings if requested."""
    if start_year is None and end_year is None:
        return da
    if 'time' not in da.coords:
        raise ValueError('Cannot use --start-year/--end-year because the data have no time coordinate.')

    start = None if start_year is None else str(start_year)
    end = None if end_year is None else str(end_year)
    return da.sel(time=slice(start, end))


def subset_lat(da: xr.DataArray, lat_range: tuple[float, float] | list[float] | None) -> xr.DataArray:
    """Subset latitude while handling ascending or descending latitude coordinates."""
    if lat_range is None:
        return da
    if 'lat' not in da.coords:
        raise ValueError('Cannot use --lat-range because the data have no lat coordinate.')

    lat_min, lat_max = lat_range
    lat = da['lat']
    if lat.size < 2 or lat[0] < lat[-1]:
        return da.sel(lat=slice(lat_min, lat_max))
    return da.sel(lat=slice(lat_max, lat_min))


def format_model_label(models) -> str:
    """Format EnvCpt model choices for output filenames."""
    return '-'.join(models)


def make_default_prefix(args: argparse.Namespace) -> str:
    """Make an output prefix that avoids collisions across common settings."""
    parts = [args.target_file.stem]
    if args.nudge_file is not None:
        parts.extend(['minus', args.nudge_file.stem])
    if args.start_year is not None or args.end_year is not None:
        start = 'start' if args.start_year is None else str(args.start_year)
        end = 'end' if args.end_year is None else str(args.end_year)
        parts.append(f'{start}-{end}')
    parts.extend([
        f'model{format_model_label(args.envcpt_models)}',
        f'max{args.max_cps}',
        f'minseg{args.minseglen}',
        'std' if args.do_std else 'nostd',
    ])
    return '_'.join(parts)


def standardize_monthly_time(da):
    import pandas as pd
    years = da['time.year'].values
    months = da['time.month'].values

    new_time = pd.to_datetime([
        f'{int(y):04d}-{int(m):02d}-01'
        for y, m in zip(years, months)
    ])

    return da.assign_coords(time=new_time)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.output_prefix or make_default_prefix(args)

    da_target = open_dataarray(args.target_file, args.target_var)
    da_target = subset_time(da_target, args.start_year, args.end_year)

    if args.nudge_file is not None:
        da_nudge = open_dataarray(args.nudge_file, args.nudge_var)
        da_nudge = subset_time(da_nudge, args.start_year, args.end_year)
        da_nudge = args.nudge_scale * da_nudge

        # Ensure that calendars match (CESM2 has noleap default)
        da_target = standardize_monthly_time(da_target)
        da_nudge = standardize_monthly_time(da_nudge)

        da_target, da_nudge = xr.align(da_target, da_nudge, join='inner')
        delta = da_target - da_nudge
    else:
        delta = da_target

    if args.land_mask is not None:
        land_mask = open_dataarray(args.land_mask, args.land_mask_var).squeeze()
        delta, land_mask = xr.align(delta, land_mask, join='inner')
        delta = delta.where(land_mask > args.land_cut)

    delta = subset_lat(delta, args.lat_range)

    eps = 1e-12
    valid_for_cp = (
        np.isfinite(delta).all('time')
        & (delta.count('time') >= 2 * args.minseglen)
        & (delta.std('time', skipna=True) > eps)
    )

    delta = delta.where(valid_for_cp)

    if not args.no_load:
        delta = delta.load()

    if args.do_std:
        delta_norm = my_utils.normalize_delta(delta)
    else:
        delta_norm = my_utils.normalize_delta(delta, do_std=False)

    # Keep the original behavior for estimating magnitudes: fit the model to
    # mean-centered, but not standard-deviation-scaled, data.
    delta_mu_norm = my_utils.normalize_delta(delta, do_std=False)

    delta_norm = delta_norm.where(valid_for_cp)
    delta_mu_norm = delta_mu_norm.where(valid_for_cp)

    if not args.no_load:
        delta_norm = delta_norm.load()
        delta_mu_norm = delta_mu_norm.load()

    cp_info_file = args.output_dir / f'ds_cp_info_{prefix}.nc'
    delta_at_cp_file = args.output_dir / f'da_delta_at_cp_{prefix}.nc'
    fitted_file = args.output_dir / f'fitted_da_{prefix}.nc'

    if cp_info_file.exists() and not args.overwrite:
        ds_cp_info = xr.open_dataset(cp_info_file)
    else:
        r_fn_envcpt = my_utils.get_R_fn_envcpt_ar1(models=args.envcpt_models)
        ds_cp_info = my_utils.return_changepoint_info(
            da=delta_norm,
            fn=r_fn_envcpt,
            max_cps=args.max_cps,
            minseglen=args.minseglen,
        )
        ds_cp_info.to_netcdf(cp_info_file)

    da_delta_at_cp, fitted_da = my_utils.get_delta_mag(ds_cp_info, delta_mu_norm)

    da_delta_at_cp.to_netcdf(delta_at_cp_file)
    fitted_da.to_netcdf(fitted_file)

    print('Wrote:')
    print(f'  {cp_info_file}')
    print(f'  {delta_at_cp_file}')
    print(f'  {fitted_file}')


if __name__ == '__main__':
    main()
