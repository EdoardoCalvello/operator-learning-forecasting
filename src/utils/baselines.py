"""Classical forecasting baselines evaluated in physical units."""
import numpy as np


def polynomial_extrapolation(x, times_x, times_y, degree, window):
    """Least-squares polynomial fit to the last `window` inputs, evaluated at forecast times.

    Each variable is fitted independently. Degree 0 with window 1 is the constant
    (persistence) baseline.

    Args:
        x: (n_traj, n_in_steps, n_vars) observed inputs in physical units
        times_x: (n_in_steps,) input times
        times_y: (n_out_steps,) forecast times
    Returns:
        (n_traj, n_out_steps, n_vars) extrapolated forecasts
    """
    if window < degree + 1:
        raise ValueError('window must contain at least degree + 1 points')
    times_x = np.asarray(times_x, dtype=np.float64)
    times_y = np.asarray(times_y, dtype=np.float64)
    # Shift so the last input is t=0 and rescale to [-1, 1] for conditioning.
    t_fit = times_x[-window:] - times_x[-1]
    t_out = times_y - times_x[-1]
    scale = max(abs(t_fit[0]), abs(t_out[-1]))
    fit = np.vander(t_fit / scale, degree + 1)
    out = np.vander(t_out / scale, degree + 1)
    # Fitting then evaluating is linear in the data: one (n_out, window) operator.
    operator = out @ np.linalg.pinv(fit)
    window_data = np.asarray(x[:, -window:, :], dtype=np.float64)
    return np.einsum('ow,nwv->nov', operator, window_data)


def relative_l2_per_trajectory(prediction, truth):
    """||prediction - truth|| / ||truth|| over time and variables, per trajectory."""
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    axes = tuple(range(1, truth.ndim))
    return np.sqrt(((prediction - truth) ** 2).sum(axis=axes) / (truth ** 2).sum(axis=axes))


def trajectory_statistics_errors(prediction, truth):
    """Errors in each trajectory's time-mean and time-variance, as in the evaluation scripts.

    Absolute differences per trajectory and variable, averaged over variables then trajectories.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    mean_error = np.abs(prediction.mean(axis=1) - truth.mean(axis=1)).mean()
    var_error = np.abs(prediction.var(axis=1) - truth.var(axis=1)).mean()
    return float(mean_error), float(var_error)
