import math
import numpy as np

from dataclasses import dataclass
from numba import njit, prange
from typing import Optional

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


# utils
## ConceptEvaluator
### collect pairs, concepts, and features
@njit(cache=True)
def collect_fpc(
    cs_sele: np.ndarray, n_fpc: int,
    p_fpc: np.ndarray, k_fpc: np.ndarray
) -> tuple[np.ndarray, np.ndarray, set, np.ndarray, np.ndarray, np.ndarray]:
    cs_sele_set = set(cs_sele)
    nk = len(k_fpc)

    # build selection mask and collect features and concepts
    mask = np.zeros(nk, dtype=np.bool_)

    # typing
    _c0 = k_fpc[0]
    _f0 = p_fpc[0, 0]

    cs_sele_out, fs_sele = {_c0}, {_f0}
    cs_sele_out.clear()
    fs_sele.clear()

    cs_no_sele, fs_no_sele = [_c0], {_f0}
    cs_no_sele.clear()
    fs_no_sele.clear()

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

### confusion matrix utils
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

#### build mapping for prediction indices
@njit(cache=True)
def build_pred_idx_map(n_features: int, fs_sele: np.ndarray) -> np.ndarray:
    pred_idx_map = np.full(n_features, -1, dtype=np.int32)
    for f_i in range(fs_sele.size):
        pred_idx_map[fs_sele[f_i]] = f_i
    return pred_idx_map


## confusion matrix
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

### only when the fpc given
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

### calculate concept scores
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

## calculate_concept_smd
@njit(parallel=True, cache=True)
def fast_eval_concept_smd(
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

        neg_check = np.zeros(neg_arr.shape[0], dtype=np.bool_)
        for c_i in range(n_cs):
            c = cs_sele[c_i]

            if not neg_check[c]:
                for f in nz_fs:
                    neg_arr[c, f] += neg_preds[c_i, f]
                    neg_sq_arr[c, f] += neg_preds_sq[c_i, f]
                ctr[c, 0] +=  neg_cnt[c_i]
                neg_check[c] = True

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

## calculate_repr_smd
@njit(parallel=True, cache=True)
def fast_eval_repr_smd(
    arr: np.ndarray, arr_sq: np.ndarray,
    curr_ptr: int,
    mol_indptr: np.ndarray, mol_data: np.ndarray,
    nz_mask: np.ndarray
) -> None:
    n_fs = nz_mask.shape[0]
    for f in prange(n_fs):
        if nz_mask[f]:
            start = mol_indptr[f]
            end = mol_indptr[f + 1]

            max_act = mol_data[start]
            for i in range(start + 1, end):
                act = mol_data[i]
                if act > max_act:
                    max_act = act

            arr[curr_ptr, f] += max_act
            arr_sq[curr_ptr, f] += max_act ** 2
