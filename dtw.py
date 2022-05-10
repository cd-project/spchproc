import numpy as np


def dtw(t: np.ndarray, s: np.ndarray, dist_list: list, warp=2):
    """

    :param t array t: (N1, M) array
    :param s array s: (N1, M) array
    :param dist_list: list of distance function apply for each N dimension vector of template
    :param warp: warp scale allowed
    :return total cost and alignment paths
    """

    r, c = len(t), len(s)
    C = np.zeros((r + 1, c + 1))
    C[0, 1:] = np.inf
    C[1:, 0] = np.inf
    for i in range(1, r + 1):
        for j in range(1, c + 1):
            C[i, j] = dist_list[i - 1](t[i - 1], s[j - 1])

    P = np.zeros((r + 1, c + 1))
    for i in range(0, r + 1):
        for j in range(0, c + 1):
            _j = max(j - 1, 0)
            cost_list = []
            for k in range(0, warp + 1):
                cost_list += [P[max(i - k, 0), _j]]

            P[i, j] = min(cost_list) + C[i, j]

    optimal_path_t, optimal_path_s = tracing_path(P, warp)
    return P[-1, -1], optimal_path_t, optimal_path_s


def tracing_path(P: np.ndarray, warp: int):
    """

    :param cost matrix P: (n_mfcc, n_fcc) array
    :param warping step warp: int
    :return: optimal aligned path for template and sample: (n_state + 2, ) array, (n_frame + 2, ) array
    """

    path_t = []
    path_s = []

    i, j = P.shape[0] - 1, P.shape[1] - 1
    path_t.insert(0, i)
    path_s.insert(0, j)
    while i > 0 or j > 0:
        next_index = np.argmin([P[max(i - k, 0), j - 1] for k in range(0, warp + 1)])

        j = max(j - 1, 0)
        i = max(i - next_index, 0)

        path_t.insert(0, i)
        path_s.insert(0, j)

    return path_t, path_s
