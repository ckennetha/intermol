import os
import tempfile
import warnings

import math
import h5py
import numpy as np

from dataclasses import dataclass
from typing import Optional
from tqdm.auto import tqdm
from numba import njit, prange
from scipy.sparse import csc_matrix

from intermol.interp.utils import h5_chunk_sorter

# dataclasses
@dataclass
class Metrics:
    conceptIdx: int
    featureIdx: int
    threshold: float
    tPos: int
    fNeg: int
    fPos: int
    tNeg: int
    precision: float
    recall: float
    f1Score: float
    tPosSub: int = 0

@dataclass
class SMDOutput:
    conceptIdx: int
    featureIdx: int
    smd: float


# core funcs
class ConceptEvaluator:
    def __init__(
        self, acts_h5_pth: str, batch_size: int = 65536
    ):
        self.acts_h5_pth = acts_h5_pth
        self.batch_size = batch_size # batch size for accessing np.memmap

    def _init_confusion_matrix(self, shape: tuple[int, int, int]) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as ntf:
            self.tmp_name = ntf.name
            self.cm = np.memmap(
                self.tmp_name, mode='w+', shape=shape, dtype=np.uint64
            )

    def _cleanup_confusion_matrix(self) -> None:
        self.cm._mmap.close()
        del self.cm
        os.unlink(self.tmp_name)

    def eval(
        self,
        samples: dict[int, dict[int, list[int]]],
        n_concepts: int,
        thresholds: float | list[float] = 0,
        fpc_map: Optional[dict[int, list[int]]] = None,
        n_fpc: Optional[int] = None
    ) -> list[Metrics]:
        # thresholds dtype check
        if isinstance(thresholds, (int, float)):
            thresholds = [thresholds]
        n_thresholds = len(thresholds)
        thresholds = np.array(thresholds)

        # open acts_h5_pth
        try:
            with h5py.File(self.acts_h5_pth, 'r') as h5f:
                chunks = h5_chunk_sorter(list(h5f.keys()))
                n_features = h5f.attrs["num_features"]

                # init confusion matrix
                use_fpc = False
                if fpc_map is None:
                    shape = (n_concepts * n_features, n_thresholds, 4)
                else:
                    use_fpc = True
                    shape = (n_concepts * n_fpc, n_thresholds, 4)

                    # build paired fpc map
                    k_fpc = np.fromiter(fpc_map.keys(), dtype=np.uint32)
                    v_fpc = list(fpc_map.values())
                    p_fpc = np.full((len(fpc_map), n_fpc), -1, dtype=np.int16)
                    for v_i, v in enumerate(v_fpc):
                        p_fpc[v_i, :len(v)] = v

                self._init_confusion_matrix(shape)

                # eval
                curr_smi = 0
                with tqdm(total=len(samples), desc="Processing samples...") as pbar:
                    for c in chunks:
                        g = h5f[c]

                        molptr = g['molptr'][:]
                        indptr = g['indptr'][:]
                        indices = g['indices'][:]
                        data = g['data'][:]

                        curr_indptr = 0
                        curr_indices = 0
                        for nt in molptr:
                            e_indptr = curr_indptr + n_features + 1
                            mol_indptr = indptr[curr_indptr:e_indptr]

                            e_indices = curr_indices + mol_indptr[-1]
                            # skip if NOT in samples
                            if curr_smi not in samples:
                                curr_indptr = e_indptr
                                curr_indices = e_indices
                                curr_smi += 1
                                continue

                            mol_indices = indices[curr_indices:e_indices]
                            mol_data = data[curr_indices:e_indices]

                            # build labels
                            ls_dict = samples[curr_smi]
                            cs_sele = np.fromiter(ls_dict.keys(), dtype=np.uint32)

                            # rebuild labels with fpc given
                            if use_fpc:
                                fs_no_sele = None
                                (
                                    p_sele, fs_sele, cs_sele,
                                    p_no_sele, fs_no_sele, cs_no_sele
                                ) = collect_fpc(cs_sele, n_fpc, p_fpc, k_fpc)

                                # update labels
                                ls_dict = {
                                    cs: ls for cs, ls in ls_dict.items() if cs in cs_sele
                                }
                                n_ls = len(ls_dict)

                                # ensure exact sequence to the labels
                                cs_sele = np.fromiter(ls_dict.keys(), dtype=np.uint32)
                                ls = list(ls_dict.values())
                            else:
                                n_ls = len(ls_dict)
                                ls = list(ls_dict.values())
                                nz_cs = np.diff(mol_indptr)
                                fs_sele = np.flatnonzero(nz_cs)

                            try:
                                rows = np.concatenate(ls)
                                cols = np.repeat(
                                    range(n_ls), [len(label) for label in ls]
                                )
                                label = np.zeros((nt, cs_sele.size), dtype=np.int8)
                                label[rows, cols] = 1

                                update_confusion_matrix(
                                    self.cm,
                                    mol_indptr, mol_indices, mol_data,
                                    nt, n_features, thresholds,
                                    cs_sele, fs_sele, label,
                                    use_fpc, p_sele
                                )
                            except ValueError as e:
                                warnings.warn(
                                    "Error updating the confusion matrix for sample "
                                    f"{curr_smi}: {str(e)}"
                                )

                            if use_fpc and (fs_no_sele is not None):
                                update_no_confusion_matrix(
                                    self.cm,
                                    mol_indptr, mol_data,
                                    nt, n_features, thresholds,
                                    cs_no_sele, fs_no_sele, p_no_sele
                                )

                            curr_indptr = e_indptr
                            curr_indices = e_indices
                            curr_smi += 1

                            pbar.update()
                            # update confusion matrix
                            self.cm.flush()

                        # early stop
                        if pbar.n == pbar.total:
                            break

            del (
                mol_indptr, mol_indices, mol_data,
                cs_sele, fs_sele, p_sele,
                cs_no_sele, fs_no_sele, p_no_sele
            )

            nr = self.cm.shape[0]
            nb = math.ceil(nr / self.batch_size)
            outs = []
            for b_i in tqdm(range(nb), leave=False):
                sb = b_i * self.batch_size
                eb = min(sb + self.batch_size, nr)

                cm_b = self.cm[sb:eb]
                rgs = (sb, eb)
                outs.extend(
                    score_concepts(
                        cm_b, thresholds, rgs, n_features, use_fpc, n_fpc,
                        is_substruct=False
                    )
                )
        finally:
            self._cleanup_confusion_matrix()

        return outs

    def eval_substructure(
        self,
        samples: dict[int, dict[int, list[int]]],
        n_concepts: int,
        thresholds: float | list[float] = 0,
        fpc_map: Optional[dict[int, list[int]]] = None,
        n_fpc: Optional[int] = None
    ) -> list[Metrics]:
        # thresholds dtype check
        if isinstance(thresholds, (int, float)):
            thresholds = [thresholds]
        n_thresholds = len(thresholds)
        thresholds = np.array(thresholds)

        # open acts_h5_pth
        try:
            with h5py.File(self.acts_h5_pth, 'r') as h5f:
                chunks = h5_chunk_sorter(list(h5f.keys()))
                n_features = h5f.attrs["num_features"]

                # init confusion matrix & tp_substruct
                use_fpc = False
                if fpc_map is None:
                    nr = n_concepts * n_features
                    shape = (nr, n_thresholds, 4)
                else:
                    use_fpc = True
                    nr = n_concepts * n_fpc
                    shape = (nr, n_thresholds, 4)

                    # build paired fpc map
                    k_fpc = np.fromiter(fpc_map.keys(), dtype=np.uint32)
                    v_fpc = list(fpc_map.values())
                    p_fpc = np.full((len(fpc_map), n_fpc), -1, dtype=np.int16)
                    for v_i, v in enumerate(v_fpc):
                        p_fpc[v_i, :len(v)] = v

                self._init_confusion_matrix(shape)
                tp_substruct = np.zeros((nr, n_thresholds), dtype=np.uint64)

                # init total substructure counter
                ctr_substruct = np.zeros(n_concepts, dtype=np.uint64)

                # eval
                curr_smi = 0
                with tqdm(total=len(samples), desc="Processing samples...") as pbar:
                    for c in chunks:
                        g = h5f[c]

                        molptr = g['molptr'][:]
                        indptr = g['indptr'][:]
                        indices = g['indices'][:]
                        data = g['data'][:]

                        curr_indptr = 0
                        curr_indices = 0
                        for nt in molptr:
                            e_indptr = curr_indptr + n_features + 1
                            mol_indptr = indptr[curr_indptr:e_indptr]

                            e_indices = curr_indices + mol_indptr[-1]
                            # skip if NOT in samples
                            if curr_smi not in samples:
                                curr_indptr = e_indptr
                                curr_indices = e_indices
                                curr_smi += 1
                                continue

                            mol_indices = indices[curr_indices:e_indices]
                            mol_data = data[curr_indices:e_indices]

                            # build labels
                            ls_dict = samples[curr_smi]
                            cs_sele = np.fromiter(ls_dict.keys(), dtype=np.uint32)

                            # rebuild labels with fpc given
                            if use_fpc:
                                fs_no_sele = None
                                (
                                    p_sele, fs_sele, cs_sele,
                                    p_no_sele, fs_no_sele, cs_no_sele
                                ) = collect_fpc(cs_sele, n_fpc, p_fpc, k_fpc)

                                # update labels
                                ls_dict = {
                                    cs: ls for cs, ls in ls_dict.items() if cs in cs_sele
                                }

                                # ensure exact sequence to the labels
                                cs_sele = np.fromiter(ls_dict.keys(), dtype=np.uint32)
                            else:
                                nz_cnt = np.diff(mol_indptr)
                                fs_sele = np.flatnonzero(nz_cnt)
                                p_sele = np.empty((1, 1), dtype=np.uint64)

                            # merge and flat labels
                            rows = []
                            cols = []
                            cols_sub = []
                            sub_idx = 0

                            ls_sub_indptr = np.zeros(cs_sele.size + 1, dtype=np.uint16)
                            ls_sub_indices = []
                            offset = 0
                            for l_i, (c, l) in enumerate(ls_dict.items()):
                                if l:
                                    ctr_substruct[c] += len(l)
                                    cols.extend([l_i] * len(l) * len(l[0]))
                                    for tk_idxs in l:
                                        rows.extend(tk_idxs)
                                        cols_sub.extend([sub_idx] * len(tk_idxs))
                                        ls_sub_indices.append(sub_idx)
                                        sub_idx += 1
                                    offset += len(l)
                                else:
                                    sub_idx += 1
                                ls_sub_indptr[l_i+1] = offset
                            ls_sub_indices = np.array(ls_sub_indices, dtype=np.uint16)

                            # run
                            try:
                                # token-level labels
                                rows = np.array(rows, dtype=np.uint16)
                                cols = np.array(cols, dtype=np.uint16)
                                label = np.zeros((nt, cs_sele.size), dtype=np.uint8)
                                label[rows, cols] = 1

                                # substructure-level labels
                                cols_sub = np.array(cols_sub, dtype=np.uint16)
                                label_sub = np.zeros((nt, sub_idx + 1), dtype=np.uint8)
                                label_sub[rows, cols_sub] = 1
                                label_sub_sum = np.sum(label_sub, axis=0)

                                update_confusion_matrix_substructure(
                                    self.cm, tp_substruct,
                                    mol_indptr, mol_indices, mol_data,
                                    map_sub_indices=ls_sub_indices,
                                    map_sub_indptr=ls_sub_indptr,
                                    nt=nt, n_features=n_features, thresholds=thresholds,
                                    cs_sele=cs_sele, fs_sele=fs_sele,
                                    label=label, label_sub=label_sub,
                                    label_sub_sum=label_sub_sum,
                                    use_fpc=use_fpc, p_sele=p_sele
                                )
                            except ValueError as e:
                                warnings.warn(
                                    "Error updating the confusion matrix for sample "
                                    f"{curr_smi}: {str(e)}"
                                )

                            if use_fpc and (fs_no_sele is not None):
                                update_no_confusion_matrix(
                                    self.cm,
                                    mol_indptr, mol_data,
                                    nt, n_features, thresholds,
                                    cs_no_sele, fs_no_sele, p_no_sele
                                )

                            curr_indptr = e_indptr
                            curr_indices = e_indices
                            curr_smi += 1

                            pbar.update()
                            self.cm.flush()

                        # early stop
                        if pbar.n == pbar.total:
                            break

            del (
                mol_indptr, mol_indices, mol_data,
                cs_sele, fs_sele, p_sele,
                cs_no_sele, fs_no_sele, p_no_sele
            )

            nr = self.cm.shape[0]
            nb = math.ceil(nr / self.batch_size)
            outs = []
            for b_i in tqdm(range(nb), leave=False):
                sb = b_i * self.batch_size
                eb = min(sb + self.batch_size, nr)

                cm_b = self.cm[sb:eb]
                rgs = (sb, eb)
                outs.extend(
                    score_concepts(
                        cm_b, thresholds, rgs, n_features, use_fpc, n_fpc,
                        is_substruct=True, tp_substruct=tp_substruct,
                        ctr_substruct=ctr_substruct
                    )
                )
        finally:
            self._cleanup_confusion_matrix()

        return outs

def calculate_smd(
    acts_h5_pth: str,
    samples: dict[int, dict[int, list[int]]],
    n_concepts: int,
    use_pooling: bool = False, # set 'True' if concept span across tokens
    eps: float = 1e-6
) -> list[SMDOutput]:
    with h5py.File(acts_h5_pth, 'r') as h5f:
        chunks = h5_chunk_sorter(list(h5f.keys()))
        n_features = h5f.attrs['num_features']

        cm_args = dict(shape=(n_concepts, n_features), dtype=np.float32)
        neg_arr, pos_arr = np.zeros(**cm_args), np.zeros(**cm_args)
        neg_sq_arr, pos_sq_arr = np.zeros(**cm_args), np.zeros(**cm_args)
        ctr = np.zeros((n_concepts, 2), dtype=np.uint32) # ctr[0]: 'neg'; ctr[1]: 'pos'

        curr_smi = 0
        with tqdm(total=len(samples), desc="Processing sample...", leave=False) as pbar:
            for c in chunks:
                g = h5f[c]

                molptr = g['molptr'][:]
                indptr = g['indptr'][:]
                indices = g['indices'][:]
                data = g['data'][:]

                curr_indptr = 0
                curr_indices = 0
                for nt in molptr:
                    e_indptr = curr_indptr + n_features + 1
                    mol_indptr = indptr[curr_indptr: e_indptr]
                    e_indices = curr_indices + mol_indptr[-1]

                    if curr_smi not in samples:
                        curr_indptr = e_indptr
                        curr_indices = e_indices
                        curr_smi += 1
                        continue

                    mol_indices = indices[curr_indices:e_indices]
                    mol_data = data[curr_indices:e_indices]

                    # rebuild activations
                    acts = csc_matrix(
                        (mol_data, mol_indices, mol_indptr),
                        shape=(nt, n_features)
                    ).toarray()

                    # labels
                    ls_dict = samples[curr_smi]
                    n_ls = len(ls_dict)
                    cs_sele = np.fromiter(ls_dict.keys(), dtype=np.uint16)

                    if use_pooling:
                        # pooled activations
                        rows = []
                        cols = []
                        n_lpr = []

                        # explode label across rows
                        rowptr = 0
                        for ls in ls_dict.values():
                            for l in ls:
                                rows.extend([rowptr] * len(l))
                                cols.extend(l)
                                rowptr += 1
                            n_lpr.append(len(ls))

                        cs_sele = np.repeat(cs_sele, n_lpr)
                        rows = np.array(rows, dtype=np.uint32)
                        cols = np.array(cols, dtype=np.uint32)
                    else:
                        # token-wise activations
                        ls = list(ls_dict.values())
                        rows = np.repeat(range(n_ls), [len(l) for l in ls])
                        cols = np.concatenate(ls)

                    # build labels
                    label = np.zeros((cs_sele.size, nt), dtype=np.float32)
                    label[rows, cols] = 1

                    # fast smd
                    fast_dense_eval_smd(
                        neg_arr, pos_arr, neg_sq_arr, pos_sq_arr,
                        ctr, nt, mol_indptr, cs_sele,
                        label, acts, use_pooling
                    )

                    curr_indptr = e_indptr
                    curr_indices = e_indices
                    curr_smi += 1
                    pbar.update()

    neg_mean = neg_arr / ctr[:, [0]]
    pos_mean = pos_arr / ctr[:, [1]]

    neg_var = (neg_sq_arr / ctr[:, [0]]) - neg_mean ** 2
    pos_var = (pos_sq_arr / ctr[:, [1]]) - pos_mean ** 2
    std = np.sqrt((pos_var + neg_var) / 2)

    smd = (pos_mean - neg_mean) / (std + eps)
    cs_sz, fs_sz = smd.shape
    outs = []
    for c_i in range(cs_sz):
        smd_c = smd[c_i, :].tolist()
        for f in range(fs_sz):
            outs.append(
                SMDOutput(
                    conceptIdx=c_i,
                    featureIdx=f,
                    smd=smd_c[f]
                )
            )
    return outs


# helpers
def score_concepts(
    cm: np.ndarray,
    thresholds: np.ndarray,
    rgs: tuple[float, float],
    n_features: int,
    use_fpc: bool = False,
    n_fpc: Optional[int] = None,
    is_substruct: bool = False,
    tp_substruct: Optional[np.ndarray] = None,
    ctr_substruct: Optional[np.ndarray] = None,
    eps: float = 1e-7
) -> list[Metrics]:
    cm = cm.astype(np.float64)
    tn, fp, fn, tp = np.moveaxis(cm, -1, 0)
    pre = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = (2 * pre * rec) / (pre + rec + eps)

    sb, eb = rgs
    outs = []

    div = n_fpc if use_fpc else n_features
    if is_substruct:
        for b_i, r_i in enumerate(range(sb, eb)):
            c_i = math.floor(r_i / div)
            f_i = r_i % div

            tp_sub = tp_substruct[b_i, :]
            cnt_sub = ctr_substruct[c_i]
            p = pre[b_i, :]
            r = tp_sub / cnt_sub
            f1_sub = (2 * p * r) / (p + r + eps)
            for thr_i, thr in enumerate(thresholds):
                if f1_sub[thr_i] > 0:
                    outs.append(
                        Metrics(
                            conceptIdx=c_i,
                            featureIdx=f_i,
                            threshold=thr,
                            tPos=int(tp[b_i, thr_i]),
                            fNeg=int(fn[b_i, thr_i]),
                            fPos=int(fp[b_i, thr_i]),
                            tNeg=int(tn[b_i, thr_i]),
                            precision=p[thr_i],
                            recall=r[thr_i],
                            f1Score=f1_sub[thr_i],
                            tPosSub=tp_sub[thr_i]
                        )
                    )
    else:
        for b_i, r_i in enumerate(range(sb, eb)):
            c_i = math.floor(r_i / div)
            f_i = r_i % div
            f1_b = f1[b_i, :]
            for thr_i, thr in enumerate(thresholds):
                if f1_b[thr_i] > 0:
                    outs.append(
                        Metrics(
                            conceptIdx=c_i,
                            featureIdx=f_i,
                            threshold=thr,
                            tPos=int(tp[b_i, thr_i]),
                            fNeg=int(fn[b_i, thr_i]),
                            fPos=int(fp[b_i, thr_i]),
                            tNeg=int(tn[b_i, thr_i]),
                            precision=pre[b_i, thr_i],
                            recall=rec[b_i, thr_i],
                            f1Score=f1_b[thr_i]
                        )
                    )
    return outs

# Numba-based helpers
## collect pairs, concepts, and features
@njit(cache=True)
def collect_fpc(
    cs_sele: np.ndarray, n_fpc: int,
    p_fpc: np.ndarray, k_fpc: np.ndarray
) -> tuple[np.ndarray, np.ndarray, set, np.ndarray, np.ndarray, np.ndarray]:
    cs_sele_set = set(cs_sele)
    nk = len(k_fpc)

    # build selection mask and collect features and concepts
    mask = np.zeros(nk, dtype=np.bool_)
    cs_sele_out, fs_sele = set(), set()
    cs_no_sele, fs_no_sele = [], set()

    for k_i in range(nk):
        c = k_fpc[k_i]
        if c in cs_sele_set:
            mask[k_i] = True
            cs_sele_out.add(c)
            fs_t = fs_sele
        else:
            cs_no_sele.append(c)
            fs_t = fs_no_sele

        for p_i in range(n_fpc):
            val = p_fpc[k_i, p_i]
            if val == -1:
                break
            fs_t.add(val)

    # split p_fpc into selected and unselected
    n_s = np.sum(mask)
    n_ns = nk - n_s
    p_sele = np.empty((n_s, n_fpc), dtype=p_fpc.dtype)
    p_no_sele = np.empty((n_ns, n_fpc), dtype=p_fpc.dtype)

    s_i, ns_i = 0, 0
    for k_i in range(nk):
        if mask[k_i]:
            p_sele[s_i] = p_fpc[k_i]
            s_i += 1
        else:
            p_no_sele[ns_i] = p_fpc[k_i]
            ns_i += 1

    fs_sele = np.array(list(fs_sele), dtype=p_fpc.dtype)
    fs_no_sele = np.array(list(fs_no_sele), dtype=p_fpc.dtype)
    cs_no_sele = np.array(cs_no_sele, dtype=cs_sele.dtype)

    return (
        p_sele, fs_sele, cs_sele_out,
        p_no_sele, fs_no_sele, cs_no_sele
    )

## compute predictions
@njit(parallel=True, cache=True)
def compute_preds(
    mol_indptr: np.ndarray, mol_indices: np.ndarray, mol_data: np.ndarray,
    nt: int, thresholds: np.ndarray, fs_sele: np.ndarray
) -> np.ndarray:
    n_thr = thresholds.size
    n_fs = fs_sele.size

    preds = np.zeros((nt, n_fs, n_thr), dtype=np.uint8)
    n_works = n_fs * n_thr
    for w_i in prange(n_works):
        f_i = w_i // n_thr
        t_i = w_i % n_thr

        f = fs_sele[f_i]
        s = mol_indptr[f]
        e = mol_indptr[f+1]

        masked_data = mol_data[s:e] > thresholds[t_i]
        masked_indices = mol_indices[s:e][masked_data]
        preds[masked_indices, f_i, t_i] = 1
    return preds

## build mapping for prediction indices
@njit(cache=True)
def build_pred_idx_map(n_features: int, fs_sele: np.ndarray) -> np.ndarray:
    pred_idx_map = np.full(n_features, -1, dtype=np.int32)
    for f_i in range(fs_sele.size):
        pred_idx_map[fs_sele[f_i]] = f_i
    return pred_idx_map

## update confusion matrix
@njit(parallel=True, cache=True)
def update_confusion_matrix(
    cm: np.ndarray,
    mol_indptr: np.ndarray, mol_indices: np.ndarray, mol_data: np.ndarray,
    nt: int, n_features: int, thresholds: np.ndarray,
    cs_sele: np.ndarray, fs_sele: np.ndarray, label: np.ndarray,
    use_fpc: bool, p_sele: np.ndarray
) -> None:
    n_thr = thresholds.size
    n_cs = cs_sele.size

    preds = compute_preds(
        mol_indptr, mol_indices, mol_data,
        nt, thresholds, fs_sele
    )
    n_works = n_cs * n_thr

    if use_fpc:
        fs_to_pred_idx = build_pred_idx_map(n_features, fs_sele)
        n_fpc = p_sele.shape[1]
        for w_i in prange(n_works):
            c_i = w_i // n_thr
            t_i = w_i % n_thr

            pred = preds[:, :, t_i]
            c = cs_sele[c_i]
            fs_at_c = p_sele[c_i]
            for f_i in range(n_fpc):
                f = fs_at_c[f_i]
                # skip padding
                if f == -1:
                    break

                pred_idx = fs_to_pred_idx[f]
                r_idx = c * n_fpc + f_i
                for tk_i in range(nt):
                    val = label[tk_i, c_i] * 2 + pred[tk_i, pred_idx]
                    cm[r_idx, t_i, val] += 1
    else:
        n_fs = fs_sele.size
        for w_i in prange(n_works):
            c_i = w_i // n_thr
            t_i = w_i % n_thr

            pred = preds[:, :, t_i]
            c = cs_sele[c_i]
            for f_i in range(n_fs):
                f = fs_sele[f_i]
                r_idx = c * n_features + f

                for tk_i in range(nt):
                    val = label[tk_i, c_i] * 2 + pred[tk_i, f_i]
                    cm[r_idx, t_i, val] += 1

## update 'no' confusion matrix (only when the fpc given)
@njit(parallel=True, cache=True)
def update_no_confusion_matrix(
    cm: np.ndarray,
    mol_indptr: np.ndarray, mol_data: np.ndarray,
    nt: int, n_features: int, thresholds: np.ndarray,
    cs_no_sele: np.ndarray, fs_no_sele: np.ndarray, p_no_sele: np.ndarray,
) -> None:
    n_thr = thresholds.size
    n_cs_no = cs_no_sele.size
    n_fs_no = fs_no_sele.size

    # create cache for positives (TP & FP)
    pos_cache = np.zeros((n_fs_no, n_thr, 2), dtype=np.uint16)
    n_works = n_fs_no * n_thr
    for w_i in prange(n_works):
        f_no_i = w_i // n_thr
        t_i = w_i % n_thr
        thr = thresholds[t_i]

        f_no = fs_no_sele[f_no_i]
        s = mol_indptr[f_no]
        e = mol_indptr[f_no + 1]

        fp_sum = np.sum(mol_data[s:e] > thr)
        pos_cache[f_no_i, t_i, 0] = nt - fp_sum
        pos_cache[f_no_i, t_i, 1] = fp_sum

    fs_no_to_pred_idx = build_pred_idx_map(n_features, fs_no_sele)
    n_fpc = p_no_sele.shape[1]
    n_works = n_cs_no * n_thr
    for w_i in prange(n_works):
        c_no_i = w_i // n_thr
        t_i = w_i % n_thr

        c_no = cs_no_sele[c_no_i]
        fs_no_at_c = p_no_sele[c_no_i]
        for f_i in range(n_fpc):
            f_no = fs_no_at_c[f_i]
            if f_no == -1:
                break

            no_pred_idx = fs_no_to_pred_idx[f_no]
            r_idx = c_no * n_fpc + f_i
            cm[r_idx, t_i, 0] += pos_cache[no_pred_idx, t_i, 0]
            cm[r_idx, t_i, 1] += pos_cache[no_pred_idx, t_i, 1]

## update confusion matrix (substructure)
@njit(parallel=True, cache=True)
def update_confusion_matrix_substructure(
    cm: np.ndarray, tp_substruct: np.ndarray,
    mol_indptr: np.ndarray, mol_indices: np.ndarray, mol_data: np.ndarray,
    map_sub_indptr: np.ndarray, map_sub_indices: np.ndarray,
    nt: int, n_features: int, thresholds: np.ndarray,
    cs_sele: np.ndarray, fs_sele: np.ndarray, label: np.ndarray,
    label_sub: np.ndarray, label_sub_sum: np.ndarray,
    use_fpc: bool, p_sele: np.ndarray
) -> None:
    n_thr = thresholds.size
    n_cs = cs_sele.size

    preds = compute_preds(
        mol_indptr, mol_indices, mol_data,
        nt, thresholds, fs_sele
    )
    n_works = n_cs * n_thr

    if use_fpc:
        fs_to_pred_idx = build_pred_idx_map(n_features, fs_sele)
        n_fpc = p_sele.shape[1]
        for w_i in prange(n_works):
            c_i = w_i // n_thr
            t_i = w_i % n_thr

            pred = preds[:, :, t_i]
            c = cs_sele[c_i]
            fs_at_c = p_sele[c_i]
            for f_i in range(n_fpc):
                f = fs_at_c[f_i]
                if f == -1:
                    break

                pred_idx = fs_to_pred_idx[f]

                # token-level
                r_idx = c * n_fpc + f_i
                for tk_i in range(nt):
                    val = label[tk_i, c_i] * 2 + pred[tk_i, pred_idx]
                    cm[r_idx, t_i, val] += 1

                # substruct-level
                s = map_sub_indptr[c_i]
                e = map_sub_indptr[c_i+1]
                for sub_i in range(s, e):
                    sub_idx = map_sub_indices[sub_i]
                    label_sum_at_sub = label_sub_sum[sub_idx]
                    if label_sum_at_sub == 0:
                        continue

                    tp_span = 0
                    for tk_i in range(nt):
                        if label_sub[tk_i, sub_idx]:
                            if pred[tk_i, pred_idx]:
                                tp_span += 1
                            else:
                                break

                    if tp_span == label_sum_at_sub:
                        tp_substruct[r_idx, t_i] += 1
    else:
        n_fs = fs_sele.size
        for w_i in prange(n_works):
            c_i = w_i // n_thr
            t_i = w_i % n_thr

            pred = preds[:, :, t_i]
            c = cs_sele[c_i]
            for f_i in range(n_fs):
                f = fs_sele[f_i]

                # token-level
                r_idx = c * n_features + f
                for tk_i in range(nt):
                    val = label[tk_i, c_i] * 2 + pred[tk_i, f_i]
                    cm[r_idx, t_i, val] += 1

                # substruct-level
                s = map_sub_indptr[c_i]
                e = map_sub_indptr[c_i+1]
                for sub_i in range(s, e):
                    sub_idx = map_sub_indices[sub_i]
                    label_sum_at_sub = label_sub_sum[sub_idx]
                    if label_sum_at_sub == 0:
                        continue

                    tp_span = 0
                    for tk_i in range(nt):
                        if label_sub[tk_i, sub_idx]:
                            if pred[tk_i, f_i]:
                                tp_span += 1
                            else:
                                break

                    if tp_span == label_sum_at_sub:
                        tp_substruct[r_idx, t_i] += 1

# fast dense eval smd
@njit(parallel=True, cache=True)
def fast_dense_eval_smd(
    neg_arr: np.ndarray, pos_arr: np.ndarray,
    neg_sq_arr: np.ndarray, pos_sq_arr: np.ndarray,
    ctr: np.ndarray, nt: int,
    mol_indptr: np.ndarray, cs_sele: np.ndarray,
    label: np.ndarray, acts: np.ndarray,
    use_pooling: bool = False
) -> None:
    nz_cs = np.diff(mol_indptr)
    nz_fs = np.flatnonzero(nz_cs)

    # preds
    preds = label @ acts
    neg_preds = acts.sum(axis=0) - preds

    # preds_sq
    acts_sq = acts * acts
    preds_sq = label @ acts_sq
    neg_preds_sq = acts_sq.sum(axis=0) - preds_sq

    # label counts
    cnt = label.sum(axis=1)
    neg_cnt = nt - cnt
    n_cs = cs_sele.size

    if use_pooling:
        # pooled SMD
        label_cnts = cnt.reshape(-1, 1)
        preds_pooled = preds / label_cnts
        preds_sq_pooled = preds_sq / label_cnts

        neg_ch = np.zeros(neg_arr.shape[0], dtype=np.bool_)
        for c_i in range(n_cs):
            c = cs_sele[c_i]

            if not neg_cnt[c]:
                for f in nz_fs:
                    neg_arr[c, f] += neg_preds[c_i, f]
                    neg_sq_arr[c, f] += neg_preds_sq[c_i, f]
                ctr[c, 0] +=  neg_cnt[c_i]
                neg_ch[c] = True

            for f in nz_fs:
                pos_arr[c, f] += preds_pooled[c_i, f]
                pos_sq_arr[c, f] += preds_sq_pooled[c_i, f]
            ctr[c, 1] += 1
    else:
        # SMD
        for c_i in prange(n_cs):
            c = cs_sele[c_i]
            for f in nz_fs:
                neg_arr[c, f] += neg_preds[c_i, f]
                pos_arr[c, f] += preds[c_i, f]

                neg_sq_arr[c, f] += neg_preds_sq[c_i, f]
                pos_sq_arr[c, f] += preds_sq[c_i, f]
            ctr[c, 0] += neg_cnt[c_i]
            ctr[c, 1] += cnt[c_i]
