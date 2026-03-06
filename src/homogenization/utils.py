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


def get_R_fn_envcpt_ar1():
    """
    Get rpy2 function object for the envcpt model

    Parameters
    ----------
    None

    Returns
    -------
    fn : rpy2 function
        Function to fit the two AR(1) models on a time series
    """

    from rpy2 import robjects as ro
    from rpy2.robjects import numpy2ri

    # Activate python - R
    numpy2ri.activate()

    ro.r('library(EnvCpt)')

    ro.r('''
    envcpt_best_model_cp_ar1_only <- function(y, minseglen = 24) {
      y2  <- as.numeric(y)

      # default outputs
      model_name <- NA_character_
      cps   <- integer(0)
      params_out <- list()

      if (length(y2) < max(4, 2*minseglen)) {
        return(list(model_name=model_name, changepoints=cps, params=params_out))
      }

      fit <- EnvCpt::envcpt(y2, minseglen=minseglen, models=c(5, 11))
      nm <- names(which.min(BIC(fit)))

      model_name <- if (length(nm)) nm[[1]] else NA_character_
      comp <- fit[[model_name]]

      # comp is usually a 'cpt' S4 object
      if (methods::is(comp, "cpt") | methods::is(comp, "cpt.reg")) {
        cps <- comp@cpts
        pe <- comp@param.est
        if (!is.null(pe)) params_out <- as.list(pe)
      }
      list(model_name=model_name, changepoints=cps, params=params_out)
    }
    ''')
    fn = ro.globalenv['envcpt_best_model_cp_ar1_only']

    return fn


def run_ts_with_r(y, fn, minseglen=24):
    """
    Run the changepoint detection on a single time series.

    Parameters
    ----------
    y : numpy.array
        The time series to check for changepoints
    fn : rpy2 function
        Function to fit the two AR(1) models on a time series

    Returns
    -------
    model : str
        The name of the selected model (via minimizing BIC)
    cp_idx : np.array
        The indices of changepoints. All will contain len(y) - 1 as last changepoint
    param_dict : dictionary
        Parameter names and associated values for each fitted segment
    """

    from rpy2 import robjects as ro
    from rpy2.rinterface_lib.sexp import NULLType

    # Fit model
    res = fn(ro.FloatVector(y), minseglen=minseglen)

    # Pull out model name
    mn = res.rx2('model_name')
    if len(mn) and mn[0] is not ro.NA_Character:
        model = str(mn[0])
    else:
        model = None

    # Get index of changepoints
    # If sole index = len(y) - 1, then no changepoint is identified
    cp_idx = np.array(list(map(int, res.rx2('changepoints'))))

    # Get parameter names (will depend on fitted model)
    params_r = res.rx2('params')
    names_r = params_r.names
    if isinstance(names_r, NULLType) or names_r is ro.NULL:
        param_names = []
    else:
        param_names = [str(n) for n in list(names_r)]  # python list of strings

    # Put values of parameters into dictionary
    # There will be a set of parameters for each segment around a changepoint
    param_dict = {}
    for p in param_names:
        param_dict[p] = params_r.rx2['%s' % p]

    return model, cp_idx, param_dict


def model_code(name): return MODEL_TO_CODE.get((name or "").lower(), -1)


def _fill_param_array(param_vals, i, j, segs, model_name, pdict):
    """Take the parameters from the dictionary and put them into an array for saving
    across all gridboxes.
    """
    m = (model_name or "").lower()

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

            # Check if time series has enough data
            if not np.sum(np.isnan(y)) < (2 * minseglen):
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

        # skip masked and places with no changepoint identifies
        if cpi == no_cp_idx or cpi < 0:
            continue

        if j == 0:
            # first changepoint
            bef = slice(0, cpi)
        else:
            bef = slice(cp_idxs[j - 1], cpi)

        if ncp == 2:
            # case of no additional changepoint
            aft = slice(cpi, ntime)
        else:
            aft = slice(cpi, cp_idxs[j + 1])

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


def add_synth_cp(da, sigma, minseglen, mu=0, seed=123):
    """
    Add a changepoint of magnitude drawn from Normal(mu, sigma) randomly to each gridcell.
    The time of the changepoint is drawn uniformly from the available time indices, excepting
    the first and last segments of length minseglen, since we cannot detect these.

    Parameters
    ----------
    da : xr.DataArray
        3D array with dims ('time', 'lat', 'lon'). Time should be monthly.
    sigma : float
        Standard deviation of the Gaussian noise to create changepoints.
    minseglen : int
        Miniumum segment length for CPs
    mu : float
        Mean of changepoints.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    da_with_cp : xr.DataArray
        Original da with artificial changepoints (time x lat x lon).
    t_idx : xr.DataArray
        Index (integer) of changepoint time for each grid cell (lat x lon).
    cp_2d : xr.DataArray
        Changepoint magnitude (lat x lon).
    """
    rng = np.random.default_rng(seed)

    ntime = da.sizes['time']
    nlat = da.sizes['lat']
    nlon = da.sizes['lon']

    minseglen = int(minseglen)  # convert in case

    # Random time index for each (lat, lon)
    t_idx = xr.DataArray(
        rng.integers(low=minseglen, high=(ntime - minseglen), size=(nlat, nlon)),
        dims=('lat', 'lon'),
        coords={'lat': da['lat'], 'lon': da['lon']}
    )

    # Convert those indices to actual time labels (datetime64)
    t_pick = da['time'].isel(time=t_idx)  # dims ('lat','lon'), dtype datetime64

    # One changepoint magnitude per grid cell
    cp_2d = xr.DataArray(
        rng.normal(loc=mu, scale=sigma, size=(nlat, nlon)),
        dims=('lat', 'lon'),
        coords={'lat': da['lat'], 'lon': da['lon']}
    )

    # Mask for all times at or after the changepoint
    # This broadcasts to (time, lat, lon):
    step_mask = da['time'] >= t_pick

    # Expand cp_2d in time and apply step mask → step change
    cp_3d = cp_2d.expand_dims(time=da['time']) * step_mask

    # Add to original
    da_with_cp = da + cp_3d

    return da_with_cp, t_idx, cp_2d


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
