#!/usr/bin/env python

import argparse
import os

import xarray as xr

import homogenization.utils as my_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description='Add synthetic changepoints to precomputed synthetic residuals and run CP detection.'
    )

    parser.add_argument(
        '--procdir',
        default='/home/data/projects/homogenization/proc',
        help='Processing directory.'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed used for the precomputed synthetic residual file.'
    )

    parser.add_argument(
        '--synth-file',
        default=None,
        help=(
            'Precomputed synthetic residual file. '
            'Default: <procdir>/synth_residuals_with_skew_seed-<seed>.nc'
        )
    )

    parser.add_argument(
        '--max-cps',
        type=int,
        default=5,
        help='Maximum number of changepoints allowed in fitted model.'
    )

    parser.add_argument(
        '--n-cp',
        type=int,
        default=1,
        help='Number of synthetic changepoints to add to each grid cell.'
    )

    parser.add_argument(
        '--minseglen',
        type=int,
        default=5 * 12,
        help='Minimum segment length in months.'
    )

    parser.add_argument(
        '--models',
        nargs='+',
        choices=['meanar1', 'trendar1'],
        default=['meanar1', 'trendar1'],
        help=(
            'EnvCpt AR(1) model(s) to allow. Default allows both '
            'meanar1 and trendar1. Use --models trendar1 to force '
            'the trend model only.'
        )
    )
    parser.add_argument(
        '--synth-cp-std',
        type=float,
        default=0.5,
        help='Standard deviation of synthetic changepoint magnitudes.'
    )

    parser.add_argument(
        '--cp-mu',
        type=float,
        default=0.0,
        help='Mean of synthetic changepoint magnitudes.'
    )

    parser.add_argument(
        '--same-cp-mag',
        action='store_true',
        help=(
            'Use the same signed changepoint magnitude for all changepoints '
            'at all grid cells. If --cp-mag-value is not provided, one shared '
            'value is drawn from Normal(cp_mu, synth_cp_std).'
        )
    )

    parser.add_argument(
        '--cp-mag-value',
        type=float,
        default=None,
        help=(
            'Fixed changepoint magnitude to use everywhere. '
            'Requires --same-cp-mag.'
        )
    )

    parser.add_argument(
        '--suffix',
        default='_with_skew',
        help='Suffix used in output filenames.'
    )

    return parser.parse_args()


def get_paths(args):
    synth_file = (
        args.synth_file
        if args.synth_file is not None
        else f'{args.procdir}/synth_residuals_with_skew_seed-{args.seed}.nc'
    )

    file_tag = (
        f'{args.suffix}_ncp-{args.n_cp}_'
        f'sigma-{args.synth_cp_std:0.1f}_seed-{args.seed}'
    )

    if args.same_cp_mag:
        file_tag += '_samecpmag'

        if args.cp_mag_value is not None:
            file_tag += f'_cpval-{args.cp_mag_value:0.2f}'
            
    default_models = ['meanar1', 'trendar1']
    if args.models != default_models:
        model_tag = '-'.join(args.models)
        file_tag += f'_models-{model_tag}'

    output_files = {
        't_idx': f'{args.procdir}/synth_cps_time{file_tag}.nc',
        'cp_mag': f'{args.procdir}/synth_cps_cpmag{file_tag}.nc',
        'cp_info': f'{args.procdir}/ds_cp_info_synth{file_tag}.nc',
        'delta_at_cp': f'{args.procdir}/da_delta_at_cp_synth{file_tag}.nc',
        'fitted_da': f'{args.procdir}/fitted_da_synth{file_tag}.nc',
    }

    return synth_file, output_files


def main():
    args = parse_args()

    if args.cp_mag_value is not None and not args.same_cp_mag:
        raise ValueError('--cp-mag-value requires --same-cp-mag')

    os.makedirs(args.procdir, exist_ok=True)

    synth_file, output_files = get_paths(args)

    da_synthetic = xr.open_dataarray(synth_file).load()

    # Ensure expected dimension order
    da_synthetic = da_synthetic.transpose('time', 'lat', 'lon')

    # Add synthetic changepoints
    da_synthetic_with_cps, t_idx, cp_mag = my_utils.add_synth_cp(
        da_synthetic,
        sigma=args.synth_cp_std,
        minseglen=args.minseglen,
        n_cp=args.n_cp,
        mu=args.cp_mu,
        seed=args.seed + 1,
        same_cp_mag=args.same_cp_mag,
        cp_mag_value=args.cp_mag_value,
    )

    # Save known synthetic changepoint inputs
    t_idx.to_netcdf(output_files['t_idx'])
    cp_mag.to_netcdf(output_files['cp_mag'])

    # Remove climatological mean and variance before CP detection
    da_synthetic_with_cps_norm = my_utils.normalize_delta(
        da_synthetic_with_cps
    )

    # Run changepoint algorithm
    r_fn_envcpt = my_utils.get_R_fn_envcpt_ar1(models=args.models)

    ds_cp_info = my_utils.return_changepoint_info(
        da=da_synthetic_with_cps_norm,
        fn=r_fn_envcpt,
        max_cps=args.max_cps,
        minseglen=args.minseglen,
    )

    ds_cp_info.to_netcdf(output_files['cp_info'])

    # Fit changepoint model to data with only climatological mean removed
    da_synthetic_with_cps_norm_mean = my_utils.normalize_delta(
        da_synthetic_with_cps,
        do_std=False,
    )

    da_delta_at_cp, fitted_da = my_utils.get_delta_mag(
        ds_cp_info,
        da_synthetic_with_cps_norm_mean,
    )

    da_delta_at_cp.to_netcdf(output_files['delta_at_cp'])
    fitted_da.to_netcdf(output_files['fitted_da'])


if __name__ == '__main__':
    main()