import numpy as np
import xarray as xr
import json


param_names = ['mean', 'slope', 'phi1', 'phi2', 'sig2']
param_idx_dict = {p: i for i, p in enumerate(param_names)}

MODEL_TO_CODE = {
    "mean": 0, "meancpt": 1,
    "meanar1": 2, "meanar2": 3,
    "meanar1cpt": 4, "meanar2cpt": 5,
    "trend": 6, "trendcpt": 7,
    "trendar1": 8, "trendar2": 9,
    "trendar1cpt": 10, "trendar2cpt": 11,
}


BETA_ORDER = {
    'mean':            ['mean'],
    'meancpt':         ['mean'],
    'meanar1':         ['mean', 'phi1'],
    'meanar2':         ['mean', 'phi1', 'phi2'],
    'meanar1cpt':      ['mean', 'phi1'],
    'meanar2cpt':      ['mean', 'phi1', 'phi2'],
    'trend':           ['mean', 'slope'],
    'trendcpt':        ['mean', 'slope'],
    'trendar1':        ['mean', 'slope', 'phi1'],
    'trendar2':        ['mean', 'slope', 'phi1', 'phi2'],
    'trendar1cpt':     ['mean', 'slope', 'phi1'],
    'trendar2cpt':     ['mean', 'slope', 'phi1', 'phi2'],
}


def normalize_delta(da, do_std=True):
    """Remove seasonality in the residual (target minus reference).

    Parameters
    ----------
    da : xr.DataArray
        Contains the residuals, time x lat x lon
    do_std : bool
        Divide by sample standard deviation by month?

    Returns
    -------
    da_norm : xr.DataArray
        Normalized residuals, time x lat x lon
    """

    da_norm = da.groupby('time.month') - da.groupby('time.month').mean()
    if do_std:
        da_norm = da_norm.groupby('time.month') / da_norm.groupby('time.month').std()

    return da_norm



def get_R_fn_envcpt_ar1(models=('meanar1', 'trendar1')):
    """
    Get rpy2 function object for the EnvCpt AR(1) model family.

    Parameters
    ----------
    models : tuple or list of str
        EnvCpt model names to allow. Options:
        - 'meanar1'  -> EnvCpt model 5
        - 'trendar1' -> EnvCpt model 11

    Returns
    -------
    fn : rpy2 function
        Function to fit the selected AR(1) EnvCpt model(s) on a time series.
    """

    from rpy2 import robjects as ro
    from rpy2.robjects import numpy2ri

    model_map = {
        'meanar1': 5,
        'trendar1': 11,
    }

    if isinstance(models, str):
        models = [models]

    invalid = [m for m in models if m not in model_map]
    if invalid:
        raise ValueError(
            f'Invalid EnvCpt model(s): {invalid}. '
            f'Valid options are: {list(model_map)}'
        )

    model_codes = [model_map[m] for m in models]
    model_codes_r = ', '.join(str(code) for code in model_codes)

    # Activate python - R
    numpy2ri.activate()

    ro.r('library(EnvCpt)')

    ro.r(f'''
    envcpt_best_model_cp_ar1_only <- function(y, minseglen = 24) {{
      y2  <- as.numeric(y)

      # default outputs
      model_name <- NA_character_
      cps   <- integer(0)
      params_out <- list()

      if (length(y2) < max(4, 2*minseglen)) {{
        return(list(model_name=model_name, changepoints=cps, params=params_out))
      }}

      fit <- EnvCpt::envcpt(y2, minseglen=minseglen, models=c({model_codes_r}))
      nm <- names(which.min(BIC(fit)))

      model_name <- if (length(nm)) nm[[1]] else NA_character_
      comp <- fit[[model_name]]

      # comp is usually a 'cpt' S4 object
      if (methods::is(comp, "cpt") | methods::is(comp, "cpt.reg")) {{
        cps <- comp@cpts
        pe <- comp@param.est
        if (!is.null(pe)) params_out <- as.list(pe)
      }}
      list(model_name=model_name, changepoints=cps, params=params_out)
    }}
    ''')

    fn = ro.globalenv['envcpt_best_model_cp_ar1_only']

    return fn


def run_ts_with_r(y, fn, minseglen=24):
    """
    Run the changepoint detection on a single time series.
    """

    from rpy2 import robjects as ro
    from rpy2.rinterface_lib.sexp import NULLType

    y = np.asarray(y, dtype=float)

    # Do not send invalid vectors to R.
    if (
        y.ndim != 1
        or y.size < 2 * minseglen
        or not np.all(np.isfinite(y))
        or np.nanstd(y) < 1e-12
    ):
        return None, np.array([], dtype=int), {}

    try:
        res = fn(ro.FloatVector(y), minseglen=minseglen)
    except Exception:
        return None, np.array([], dtype=int), {}

    # Pull out model name
    mn = res.rx2('model_name')
    if len(mn) and mn[0] is not ro.NA_Character:
        model = str(mn[0])
    else:
        model = None

    # Get index of changepoints
    cp_idx = np.array(list(map(int, res.rx2('changepoints'))), dtype=int)

    # Get parameter names
    params_r = res.rx2('params')
    names_r = params_r.names
    if isinstance(names_r, NULLType) or names_r is ro.NULL:
        param_names_r = []
    else:
        param_names_r = [str(n) for n in list(names_r)]

    # Put values of parameters into dictionary
    param_dict = {}
    for p in param_names_r:
        param_dict[p] = params_r.rx2(p)

    return model, cp_idx, param_dict


def model_code(name): return MODEL_TO_CODE.get((name or "").lower(), -1)


def _fill_param_array(param_vals, i, j, segs, model_name, pdict):
    """Take the parameters from the dictionary and put them into an array for saving
    across all gridboxes.
    """
    m = (model_name or "").lower()


    # beta: contains mean or trend, and ar(1) information
    if 'beta' in pdict and np.asarray(pdict['beta']).size:
        B = np.asarray(pdict['beta'], dtype=float)
        B = B.reshape((B.shape[0], -1))  # (segments, ncols)
        col_names = BETA_ORDER.get(m)

        if col_names is not None:
            rows = min(segs, B.shape[0])
            for c, pname in enumerate(col_names):
                if c >= B.shape[1]:
                    continue
                pi = param_idx_dict[pname]
                param_vals[pi, :rows, i, j] = B[:rows, c]

    # variance (sig2) - one value per segment
    if 'sig2' in pdict and np.asarray(pdict['sig2']).size:
        v = np.asarray(pdict['sig2'], dtype=float).ravel()
        rows = min(segs, v.size)
        param_vals[param_idx_dict['sig2'], :rows, i, j] = v[:rows]
    """ TMP
    # beta: contains mean or trend, and ar(1) information
    if 'beta' in pdict and pdict['beta'].size:
        B = np.asarray(pdict['beta'], dtype=float)
        B = B.reshape((B.shape[0], -1))  # (segments, ncols)
        col_names = BETA_ORDER.get(m)

        rows = min(segs, B.shape[0])
        for c, pname in enumerate(col_names):
            pi = param_idx_dict[pname]
            param_vals[pi, :rows, i, j] = B[:rows, c]

    # variance (sig2) - one value per segment
    if 'sig2':
        v = np.asarray(pdict['sig2'], dtype=float).ravel()
        rows = min(segs, v.size)
        param_vals[param_idx_dict['sig2'], :rows, i, j] = v[:rows]
    """

def return_changepoint_info(da, fn, max_cps=5, minseglen=24):
    """
    For a given dataarray, fit the changepoint model at all non-NaN locations.
    Then return information about the model fits.

    Parameters
    ----------
    da : xr.DataArray
        The data (time x lat x lon) to identify changepoints (in time) in.
    fn :  rpy2 function
        Function to fit the two AR(1) models on a time series
    max_cps : int
        The maximum numer of changepoints allowed
    minseglen : int
        The minimum segment length between changepoints

    Returns
    -------


    """
    nlat, nlon = len(da['lat']), len(da['lon'])
    max_segs = max_cps + 1

    # allocate arrays to save fits
    model_code_arr = np.full((nlat, nlon), -1, dtype=np.int16)
    cp_count_arr = np.zeros((nlat, nlon), dtype=np.int16)
    cp_index_arr = np.full((max_cps, nlat, nlon), -1, dtype=np.int32)
    cp_time_arr = np.full((max_cps, nlat, nlon), np.datetime64('NaT'), dtype='datetime64[ns]')
    param_vals = np.full((len(param_names), max_segs, nlat, nlon), np.nan, dtype=float)

    # iterate through gridboxes
    for i in range(nlat):
        for j in range(nlon):

            # Pull out time series to check
            y = np.asarray(da[:, i, j].values, dtype=float)

            # Check if time series is safe to send to R
            if (
                y.ndim != 1
                or y.size < 2 * minseglen
                or not np.all(np.isfinite(y))
                or np.nanstd(y) < 1e-12
            ):
                continue

            # Fit model from EnvCpt
            model, cp_idx0, pdict = run_ts_with_r(y, fn, minseglen=minseglen)

            # Save numerical model code
            model_code_arr[i, j] = model_code(model)

            # Save changepoint index and time
            k = min(len(cp_idx0), max_cps)
            if k:
                cp_index_arr[:k, i, j] = cp_idx0[:k]
                cp_time_arr[:k,  i, j] = da['time'][cp_idx0[:k]]
            cp_count_arr[i, j] = k

            # segments (= cp_count + 1), then params
            segs = min(k + 1, max_segs)
            _fill_param_array(param_vals, i, j, segs, model, pdict)

    # Compile dataset for various parameter values and changepoint locations
    ds = xr.Dataset(
        data_vars=dict(
            model_code=(('lat', 'lon'), model_code_arr),
            cp_count=(('lat', 'lon'), cp_count_arr),
            cp_index=(('cp', 'lat', 'lon'), cp_index_arr),
            cp_time=(('cp', 'lat', 'lon'), cp_time_arr),
            params=(('param', 'seg', 'lat', 'lon'), param_vals),
        ),
        coords=dict(
            lat=da['lat'], lon=da['lon'],
            cp=np.arange(max_cps), seg=np.arange(max_segs),
            param=np.array(param_names, dtype=object),
        ),
        attrs=dict(
            model_code_map=json.dumps(MODEL_TO_CODE),
            param_semantics=json.dumps({
                'mean': 'segment mean (or intercept)',
                'slope': 'trend slope per segment',
                'phi1': 'AR(1) coefficient',
                'phi2': 'AR(2) coefficient',
                'sig2': 'segment variance'
            })
        )
    )
    return ds


def _linfit(x, y):
    """Fit y = a + b * x after removing NaNs"""
    from scipy.stats import linregress
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan, np.nan
    slope, intercept, *_ = linregress(x[m], y[m])

    return intercept, slope


def linfit_ar1_glsar(x, y, max_iter=50, rtol=1e-8):
    """
    Feasible GLS with AR(1) errors via statsmodels.GLSAR (iterative fit).
    Note: GLSAR uses a Cochrane–Orcutt-style whitening (effectively drops the first obs).
    Returns: a, b, rho
    """
    import statsmodels.api as sm
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        X = sm.add_constant(x, has_constant='add')
        a, b = np.linalg.lstsq(X, y, rcond=None)[0]
        return float(a), float(b), 0.0

    X = sm.add_constant(x, has_constant='add')
    model = sm.GLSAR(y, X, rho=1)
    res = model.iterative_fit(maxiter=max_iter, rtol=rtol)

    a, b = res.params
    rho = float(np.atleast_1d(model.rho)[0])
    return float(a), float(b), rho


def fit_cp_model(ts, no_cp_idx, cp_idxs, model_code):
    """
    Parameters
    ----------
    ts : np.array
        Time series of residuals
    cp_idxs : np.array
        Indices associated with changepoints
    model_code : int
        Code for model (mean vs trend)

    Returns
    -------
    fitted : np.array
        The fitted estimates of the segments, length ntime
    delta_at_cp : np.array
        The change at each change point
    """

    ts = np.asarray(ts)
    cp_idxs = np.asarray(cp_idxs, dtype=int)

    ntime = ts.shape[0]
    ncp = cp_idxs.shape[0]

    fitted = np.full((ntime,), np.nan, dtype=float)
    delta_at_cp = np.full((ncp,), np.nan, dtype=float)

    t = np.arange(ntime, dtype=float)
    t = t - np.mean(t)

    for j in range(ncp):  # loop through changepoints

        cpi = cp_idxs[j]
        
        # skip masked and places with no changepoint identified
        if cpi == no_cp_idx or cpi < 0:
            continue

        if j == 0:
            bef = slice(0, cpi)
        else:
            bef = slice(cp_idxs[j - 1], cpi)

        # The next cp can be the end-of-record marker or padding.
        # In those cases, fit the "after" segment through the end of ts.
        if j == (ncp - 1):
            aft = slice(cpi, ntime)
        else:
            next_cpi = cp_idxs[j + 1]

            if next_cpi == no_cp_idx or next_cpi < 0:
                aft = slice(cpi, ntime)
            else:
                aft = slice(cpi, next_cpi)
    
        if model_code == 4:
            # constant mean before/after
            m1 = np.nanmean(ts[bef])
            m2 = np.nanmean(ts[aft])

            fitted[bef] = m1
            fitted[aft] = m2
            delta_at_cp[j] = m2 - m1

        elif model_code == 10:

            a1, b1, rho1 = linfit_ar1_glsar(t[bef], ts[bef])
            a2, b2, rho2 = linfit_ar1_glsar(t[aft], ts[aft])

            fitted[bef] = a1 + b1 * t[bef]
            fitted[aft] = a2 + b2 * t[aft]

            y1_end = fitted[bef][-1]
            y2_start = fitted[aft][0]
            delta_at_cp[j] = y2_start - y1_end

            # separate linear trends before/after
            # a1, b1 = _linfit(t[bef], ts[bef])
            # a2, b2 = _linfit(t[aft], ts[aft])
            # fitted[bef] = a1 + b1 * t[bef]
            # fitted[aft] = a2 + b2 * t[aft]

            # y1_end = (a1 + b1 * t[bef])[-1]
            # y2_start = (a2 + b2 * t[aft])[0]
            # delta_at_cp[j] = y2_start - y1_end

        else:
            # unknown code
            continue

    return fitted, delta_at_cp


def get_delta_mag(ds_cp_info, delta_norm):

    # Max value = len(ts) - 1 -- if only cp, then no cp
    no_cp_idx = ds_cp_info['cp_index'].max().values

    # Vectorize across lat/lon
    fitted_da, da_delta_at_cp = xr.apply_ufunc(
        fit_cp_model,
        delta_norm,
        no_cp_idx,
        ds_cp_info['cp_index'],
        ds_cp_info['model_code'],
        input_core_dims=[['time'], [], ['cp'], []],
        output_core_dims=[['time'], ['cp']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float, float]
    )

    fitted_da = fitted_da.rename('fitted').assign_coords(time=delta_norm['time'])
    da_delta_at_cp = da_delta_at_cp.rename('delta').assign_coords(cp=ds_cp_info['cp'])

    return da_delta_at_cp, fitted_da


def add_synth_cp(
    da, sigma, minseglen, n_cp=1, mu=0, seed=123,
    same_cp_mag=False, cp_mag_value=None
):
    """
    Add user-specified number of changepoints to each grid cell.

    Changepoint times are drawn randomly for each grid cell subject to all
    segments having length at least minseglen.

    By default, each changepoint magnitude is drawn independently from
    Normal(mu, sigma). If same_cp_mag=True, all changepoints across all grid
    cells have the same magnitude. If cp_mag_value is provided, that value is
    used; otherwise one magnitude is drawn from Normal(mu, sigma).

    Parameters
    ----------
    da : xr.DataArray
        3D array with dims ('time', 'lat', 'lon'). Time should be monthly.
    sigma : float
        Standard deviation of the Gaussian distribution for changepoint magnitudes.
    minseglen : int
        Minimum segment length between changepoints and at the endpoints.
    n_cp : int
        Number of changepoints to add per grid cell.
    mu : float
        Mean of changepoint magnitudes.
    seed : int
        Random seed for reproducibility.
    same_cp_mag : bool
        If True, all changepoints have the same signed magnitude.
    cp_mag_value : float or None
        If provided with same_cp_mag=True, use this magnitude for all changepoints.
        If None, draw one shared magnitude from Normal(mu, sigma).

    Returns
    -------
    da_with_cp : xr.DataArray
        Original da with artificial changepoints added (time x lat x lon).
    t_idx : xr.DataArray
        Integer changepoint indices with dims ('cp', 'lat', 'lon').
    cp_mag : xr.DataArray
        Changepoint magnitudes with dims ('cp', 'lat', 'lon').
    """
    rng = np.random.default_rng(seed)

    ntime = da.sizes['time']
    nlat = da.sizes['lat']
    nlon = da.sizes['lon']

    minseglen = int(minseglen)
    n_cp = int(n_cp)

    if n_cp < 0:
        raise ValueError('n_cp must be non-negative')

    if n_cp == 0:
        t_idx = xr.DataArray(
            np.empty((0, nlat, nlon), dtype=int),
            dims=('cp', 'lat', 'lon'),
            coords={'cp': [], 'lat': da['lat'], 'lon': da['lon']}
        )

        cp_mag = xr.DataArray(
            np.empty((0, nlat, nlon)),
            dims=('cp', 'lat', 'lon'),
            coords={'cp': [], 'lat': da['lat'], 'lon': da['lon']}
        )

        return da.copy(), t_idx, cp_mag

    if ntime < (n_cp + 1) * minseglen:
        raise ValueError(
            f'Not enough time points for {n_cp} changepoints with '
            f'minseglen={minseglen}. Need at least {(n_cp + 1) * minseglen}, '
            f'but got ntime={ntime}.'
        )

    # Draw changepoint indices independently for each grid cell.
    # Validity requires:
    #   t1 >= minseglen
    #   t2 - t1 >= minseglen
    #   ...
    #   ntime - t_last >= minseglen
    t_idx_vals = np.empty((n_cp, nlat, nlon), dtype=int)

    extra = ntime - (n_cp + 1) * minseglen

    for i in range(nlat):
        for j in range(nlon):
            # Randomly distribute the extra time points across segments.
            # There are n_cp + 1 segments around n_cp changepoints.
            cuts = np.sort(rng.choice(extra + n_cp, size=n_cp, replace=False))
            extras = np.diff(np.r_[-1, cuts, extra + n_cp]) - 1

            seg_lengths = minseglen + extras
            t_idx_vals[:, i, j] = np.cumsum(seg_lengths)[:-1]

    t_idx = xr.DataArray(
        t_idx_vals,
        dims=('cp', 'lat', 'lon'),
        coords={
            'cp': np.arange(n_cp),
            'lat': da['lat'],
            'lon': da['lon']
        }
    )

    # Changepoint magnitudes
    if same_cp_mag:
        if cp_mag_value is None:
            shared_mag = rng.normal(loc=mu, scale=sigma)
        else:
            shared_mag = cp_mag_value

        cp_mag_vals = np.full((n_cp, nlat, nlon), shared_mag)

    else:
        cp_mag_vals = rng.normal(
            loc=mu,
            scale=sigma,
            size=(n_cp, nlat, nlon)
        )

    cp_mag = xr.DataArray(
        cp_mag_vals,
        dims=('cp', 'lat', 'lon'),
        coords={
            'cp': np.arange(n_cp),
            'lat': da['lat'],
            'lon': da['lon']
        }
    )

    # Convert changepoint indices to datetime labels
    t_pick = da['time'].isel(time=t_idx)

    # Broadcast to ('time', 'cp', 'lat', 'lon')
    step_mask = da['time'] >= t_pick

    # Sum contributions from all changepoints
    cp_signal = (cp_mag * step_mask).sum('cp')

    # Add to original
    da_with_cp = da + cp_signal

    return da_with_cp, t_idx, cp_mag


def make_synthetic_ts(time, trend_per_year, phi, sigma_monthly):
    """
    Create a synthetic monthly time series with:
      - linear trend (per year)
      - AR(1) structure with coefficient phi
      - month-of-year–dependent std dev for the *process* (not innovations)

    Parameters
    ----------
    time : xarray.DataArray or pandas.DatetimeIndex
        Monthly time coordinate.
    trend_per_year : float
        Linear trend in units per year.
    sigma_monthly : array-like or xarray.DataArray
        Length-12 array of target std devs for each calendar month (Jan..Dec).
        If DataArray, it should have a 'month' coord from 1..12.
    phi : float
        AR(1) coefficient.

    Returns
    -------
    series : numpt.array
        Synthetic time series
    """

    # Ensure time is a DataArray
    if not isinstance(time, xr.DataArray):
        time = xr.DataArray(time, dims='time', name='time')

    n = time.size

    # Handle sigma_monthly input
    if isinstance(sigma_monthly, xr.DataArray):
        sig_month = sigma_monthly
    else:
        sig_month = xr.DataArray(
            np.asarray(sigma_monthly, dtype='float64'),
            dims='month',
            coords={'month': np.arange(1, 13)}
        )

    if sig_month.sizes['month'] != 12:
        raise ValueError('sigma_monthly must have 12 values (one for each month).')

    # Get month-of-year for each time and map std devs
    month_of_year = time.dt.month
    sigma_t = sig_month.sel(month=month_of_year).values  # target process std dev at each time

    # AR(1) with time-varying innovation std so that
    # Var(X_t | month) ≈ sigma_t^2  ⇒  sigma_eps_t = sigma_t * sqrt(1 - phi^2)
    sigma_eps_t = sigma_t * np.sqrt(1.0 - phi**2)

    eps = np.random.normal(size=n)

    x = np.zeros(n, dtype='float64')

    # Initialize x[0] ~ N(0, sigma_0^2)
    x[0] = sigma_t[0] * eps[0]

    for t in range(1, n):
        x[t] = phi * x[t-1] + sigma_eps_t[t] * eps[t]

    # Add in the time trend (in /year units)
    time_years = (time.dt.year + (time.dt.month - 0.5) / 12).astype('float64')
    years_anom = time_years - np.mean(time_years)

    trend = trend_per_year * years_anom

    series = x + trend

    return series
