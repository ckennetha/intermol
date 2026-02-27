import re
import numpy as np

from numba import njit

# chunk sorter for h5py data chunks
def h5_chunk_sorter(h5_keys: list[str]):
    return sorted(h5_keys, key=lambda gn: int(re.search(r'\d+', gn).group()))


# Numba-based
## fast Spearman implementation
@njit(cache=True)
def _fast_rankdata(arr: np.ndarray) -> np.ndarray: # 'average' tied
    n = len(arr)
    sorter = np.argsort(arr)
    arr_sorted = arr[sorter]

    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and arr_sorted[j] == arr_sorted[i]:
            j += 1

        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[sorter[k]] = avg_rank

        i = j
    return ranks

@njit(cache=True)
def fast_spearmanr(y1: np.ndarray, y2: np.ndarray) -> float:
    n = len(y1)
    y1_rank = _fast_rankdata(y1)
    y2_rank = _fast_rankdata(y2)
    d_sum = np.sum((y1_rank - y2_rank) ** 2)
    rho = 1 - (6 * d_sum / (n * (n * n - 1)))
    return rho

# fast sparse pooling (both max and mean pooling)
@njit(cache=True)
def fast_sparse_pooling(
    max_arr: np.ndarray, mean_arr: np.ndarray,
    max_sq_arr: np.ndarray, mean_sq_arr: np.ndarray,
    state_smi: int, nt: int, nz_mask: np.ndarray,
    mol_indptr: np.ndarray, mol_data: np.ndarray,
    use_squared: bool = False
) -> None:
    nz_fs = nz_mask.shape[0]
    for f in range(nz_fs):
        if nz_mask[f]:
            s = mol_indptr[f]
            e = mol_indptr[f+1]

            max_act = mol_data[s]
            sum_act = mol_data[s]
            for i in range(s+1, e):
                act = mol_data[i]
                if act > max_act:
                    max_act = act
                sum_act += act
            mean_act = sum_act / nt # sequence-wise

            max_arr[state_smi, f] = max_act
            mean_arr[state_smi, f] = mean_act

            if use_squared:
                max_sq_arr[state_smi, f] += max_act ** 2
                mean_sq_arr[state_smi, f] += mean_act ** 2
