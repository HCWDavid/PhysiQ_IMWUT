import numpy as np


def expand_variables_to_segments(v, Nt):
    """ expands contextual variables v, by repeating each instance as specified in Nt """
    N_v = len(np.atleast_1d(v[0]))
    print(v.shape)
    res = []

    for i in np.arange(len(v)):
        res.append(np.full((Nt[i], N_v), v[i]))

    return np.concatenate(res, axis=0)


def check_ts_data(X, y=None):
    """
    Checks time series data is good. If not raises value error.

    Parameters
    ----------
    X : array-like, shape [n_series, ...]
       Time series data and (optionally) contextual data
       
    Returns
    -------
    ts_target : bool
        target (y) is a time series      
        
    """

    if y is not None:
        Nx = len(X)
        Ny = len(y)

        if Nx != Ny:
            raise ValueError(
                "Number of time series different in X (%d) and y (%d)" %
                (Nx, Ny)
            )

        Xt = X
        Ntx = np.array([len(Xt[i]) for i in np.arange(Nx)])
        Nty = np.array([len(np.atleast_1d(y[i])) for i in np.arange(Nx)])

        if np.count_nonzero(Nty == 1) == Nx:  # all targets are single values
            return False
        elif np.count_nonzero(Nty == Ntx) == Nx:  # y is a time series
            return True
        elif np.count_nonzero(
            Nty == Nty[0]
        ) == Nx:  # target vector (eg multilabel or onehot)
            return False
        else:
            raise ValueError(
                "Invalid time series lengths.\n"
                "Ns: ", Nx, "Ntx: ", Ntx, "Nty: ", Nty
            )


def transformTimeToSegments(X, y=None, width=100, step=None, order='F'):
    ts_target = check_ts_data(X, y)
    Xt, Nt = _segmentX(X, width, step, order)

    yt = y

    if yt is not None:
        yt = _segmentY(y, Nt, ts_target)

    return Xt, yt


def transformTimeToSegments_batch(X, y=None, width=100, step=None, order='F'):
    ts_target = check_ts_data(X, y)
    Xt, Nt = _segmentX_batch(X, width, step, order)

    yt = y

    if yt is not None:
        yt = _segmentY(y, Nt, ts_target)

    return Xt, yt


def _segmentX_batch(X, width=100, step=None, order='F'):
    Xt = X
    N = len(Xt)  # number of time series

    # print(f"number is {N}")
    if Xt[0].ndim > 1:
        # print("_segmentX: dimension greater than 1")
        Xt = np.array(
            [sliding_tensor(Xt[i], width, step, order) for i in np.arange(N)]
        )
    else:
        Xt = np.array(
            [sliding_window(Xt[i], width, step, order) for i in np.arange(N)]
        )

    Nt = [len(Xt[i]) for i in np.arange(len(Xt))]
    # print(np.sum(Nt))
    return Xt, Nt


def _segmentX(X, width=100, step=None, order='F'):
    Xt = X
    N = len(Xt)  # number of time series

    # print(f"number is {N}")
    if Xt[0].ndim > 1:
        # print("_segmentX: dimension greater than 1")
        Xt = np.array(
            [sliding_tensor(Xt[i], width, step, order) for i in np.arange(N)]
        )
    else:
        Xt = np.array(
            [sliding_window(Xt[i], width, step, order) for i in np.arange(N)]
        )

    Nt = [len(Xt[i]) for i in np.arange(len(Xt))]
    Xt = np.concatenate(Xt)
    # print(np.sum(Nt))
    return Xt, Nt


def _segmentY(y, Nt, ts_target=False):
    if ts_target:
        yt = None
        # yt = np.array([sliding_window(y[i], self.width, self._step, self.order)
        #                for i in np.arange(len(y))])
        # yt = np.concatenate(yt)
        # yt = self.y_func(yt)
    else:
        yt = expand_variables_to_segments(y, Nt)
        if yt.ndim > 1 and yt.shape[1] == 1:
            yt = yt.ravel()
    return yt


def sliding_tensor(mv_time_series, width, step, order='F'):
    """
    segments multivariate time series with sliding window

    Parameters
    ----------
    mv_time_series : array like shape [n_samples, n_variables]
        multivariate time series or sequence
    width : int > 0
        segment width in samples
    step : int > 0
        stepsize for sliding in samples

    Returns
    -------
    data : array like shape [n_segments, width, n_variables]
        segmented multivariate time series data
    """
    #     print("sliding_tensor")
    D = mv_time_series.shape[1]
    data = []
    for j in range(D):
        data.append(sliding_window(mv_time_series[:, j], width, step, order))


#     data = [sliding_window(mv_time_series[:, j], width, step, order) for j in range(D)]

    return np.stack(data, axis=2)


def sliding_window(time_series, width, step, order='F'):
    """
    Segments univariate time series with sliding window

    Parameters
    ----------
    time_series : array like shape [n_samples]
        time series or sequence
    width : int > 0
        segment width in samples
    step : int > 0
        stepsize for sliding in samples

    Returns
    -------
    w : array like shape [n_segments, width]
        resampled time series segments
    """
    #     print("sliding_window")
    res = []
    for i in range(0, width):
        # each element in the res is the length of ( total_length - width + 1)
        res.append(time_series[i:(1 + i - width or None):step])
    """
    example: [1,2,3,4,5,6,7,8,9,0]
    step=0, width=5:
    [[1, 2, 3, 4, 5, 6],
     [2, 3, 4, 5, 6, 7],
     [3, 4, 5, 6, 7, 8],
     [4, 5, 6, 7, 8, 9],
     [5, 6, 7, 8, 9, 0]]
    """

    #w = np.hstack([time_series[i:1 + i - width or None:step] for i in range(0, width)])
    w = np.hstack(res)  # stack as one dimensional np array

    result = w.reshape(
        (int(w.shape[0] / width), width), order='F'
    )  # Fortran-like index ordering
    if order == 'F':
        return result
    else:
        return np.ascontiguousarray(result)
