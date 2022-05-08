import numpy as np


def dtw(t, s, dist_list, warp=1):
    """

    :param t array t: N1*M array
    :param s array s: N1*M array
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
            cost_list = [P[i, _j]]
            for k in range(1, warp + 1):
                cost_list += [P[max(i - k, 0), _j], P[max(i - k, 0), j]]

            P[i, j] = min(cost_list) + C[i, j]

    optimal_path_t, optimal_path_s = tracing_path(P[1:, 1:], warp)
    return P[-1, -1], optimal_path_t, optimal_path_s


def tracing_path(P, warp):
    path_i = []
    path_j = []

    i, j = P.shape[0] - 1, P.shape[1] - 1
    path_i.insert(0, i)
    path_j.insert(0, j)
    while i > 0 or j > 0:
        next_index_1 = np.argmin([P[max(i - k, 0), j - 1] for k in range(1, warp + 1)])
        next_index_2 = np.argmin([P[max(i - k, 0), j] for k in range(1, warp + 1)])

        j_1 = max(j - 1, 0)
        i_1 = max(i - next_index_1 - 1, 0)

        j_2 = j
        i_2 = max(i - next_index_2 - 1, 0)
        if P[i_1, j_1] <= P[i_2, j_2]:
            i, j = i_1, j_1
        else:
            i, j = i_2, j_2

        path_i.insert(0, i)
        path_j.insert(0, j)

    return path_i, path_j
