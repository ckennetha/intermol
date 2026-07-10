import os
import tempfile
import warnings

import math
import h5py
import numpy as np

from typing import Optional
from tqdm.auto import tqdm
from scipy.sparse import csc_matrix

from intermol.interp.utils import h5_chunk_sorter
from intermol.interp.eval_utils import *

# data helpers (by Claude Opus 4.8)
## fetch i-th data in generator
def _peek_sample(state):
    return state[0]

## move forward to (i+1)-th data in generator
def _advance_sample(gen, state):
    state[0] = next(gen, None)


# core funcs
class ConceptEvaluator:
    def __init__(self, acts_h5_path: str, batch_size: int = 65536):
        self.acts_h5_path = acts_h5_path
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
        samples,
        n_samples: int,
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

        # open acts_h5_path
        try:
            with h5py.File(self.acts_h5_path, 'r') as h5f:
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
                _state = [None]
                _advance_sample(samples, _state)
                with tqdm(total=n_samples, desc="Processing samples...") as pbar:
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
                            _curr = _peek_sample(_state)
                            if (_curr is None) or (_curr[0] != curr_smi):
                                curr_indptr = e_indptr
                                curr_indices = e_indices
                                curr_smi += 1
                                continue

                            mol_indices = indices[curr_indices:e_indices]
                            mol_data = data[curr_indices:e_indices]

                            # build labels
                            ls_dict = _curr[1]
                            _advance_sample(samples, _state)

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

                                p_sele = np.empty((1, 1), dtype=np.int16)

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

                        # early stop
                        if pbar.n == pbar.total:
                            break

            del (
                mol_indptr, mol_indices, mol_data,
                cs_sele, fs_sele, p_sele
            )

            self.cm.flush()

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
        samples,
        n_samples: int,
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

        # open acts_h5_path
        try:
            with h5py.File(self.acts_h5_path, 'r') as h5f:
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
                _state = [None]
                _advance_sample(samples, _state)
                with tqdm(total=n_samples, desc="Processing samples...") as pbar:
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
                            _curr = _peek_sample(_state)
                            if (_curr is None) or (_curr[0] != curr_smi):
                                curr_indptr = e_indptr
                                curr_indices = e_indices
                                curr_smi += 1
                                continue

                            mol_indices = indices[curr_indices:e_indices]
                            mol_data = data[curr_indices:e_indices]

                            # build labels
                            ls_dict = _curr[1]
                            _advance_sample(samples, _state)

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
                                p_sele = np.empty((1, 1), dtype=np.int16)

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

                        # early stop
                        if pbar.n == pbar.total:
                            break

            del (
                mol_indptr, mol_indices, mol_data,
                cs_sele, fs_sele, p_sele
            )

            self.cm.flush()

            nr = self.cm.shape[0]
            nb = math.ceil(nr / self.batch_size)
            outs = []
            for b_i in tqdm(range(nb), leave=False):
                sb = b_i * self.batch_size
                eb = min(sb + self.batch_size, nr)

                cm_b = self.cm[sb:eb]
                tp_substruct_b = tp_substruct[sb:eb]
                rgs = (sb, eb)
                outs.extend(
                    score_concepts(
                        cm_b, thresholds, rgs, n_features, use_fpc, n_fpc,
                        is_substruct=True, tp_substruct=tp_substruct_b,
                        ctr_substruct=ctr_substruct
                    )
                )
        finally:
            self._cleanup_confusion_matrix()

        return outs

## to prefilter latents for latent-concept association
def calculate_concept_smd(
    acts_h5_path: str,
    samples,
    n_samples: int,
    n_concepts: int,
    use_pooling: bool = False, # set 'True' if concept span across tokens
    eps: float = 1e-6
) -> list[SMDConceptOutput]:
    with h5py.File(acts_h5_path, 'r') as h5f:
        chunks = h5_chunk_sorter(list(h5f.keys()))
        n_features = h5f.attrs['num_features']

        cm_args = dict(shape=(n_concepts, n_features), dtype=np.float32)
        neg_arr, pos_arr = np.zeros(**cm_args), np.zeros(**cm_args)
        neg_sq_arr, pos_sq_arr = np.zeros(**cm_args), np.zeros(**cm_args)

        # ctr[:, 0]: 'neg'; ctr[:, 1]: 'pos'
        ctr = np.zeros((n_concepts, 2), dtype=np.uint32)

        curr_smi = 0
        _state = [None]
        _advance_sample(samples, _state)
        with tqdm(total=n_samples, desc="Processing sample...", leave=False) as pbar:
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

                    _curr = _peek_sample(_state)
                    if (_curr is not None) and (_curr[0] == curr_smi):
                        mol_indices = indices[curr_indices:e_indices]
                        mol_data = data[curr_indices:e_indices]

                        # rebuild activations
                        acts = csc_matrix(
                            (mol_data, mol_indices, mol_indptr),
                            shape=(nt, n_features)
                        ).toarray()

                        # labels
                        ls_dict = _curr[1]
                        _advance_sample(samples, _state)

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
                        fast_eval_concept_smd(
                            neg_arr, pos_arr, neg_sq_arr, pos_sq_arr,
                            ctr, nt, mol_indptr, cs_sele,
                            label, acts, use_pooling
                        )

                        pbar.update()

                    curr_indptr = e_indptr
                    curr_indices = e_indices
                    curr_smi += 1

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
                SMDConceptOutput(
                    conceptIdx=c_i,
                    featureIdx=f,
                    smd=smd_c[f]
                )
            )
    return outs

## to measure latent effect size across string representations
def calculate_repr_smd(
    ref_acts_h5_path: str,
    ref_sample_map: dict[int, int],
    alt_acts_h5_path: str,
    alt_end_map: list[int],
    alt_group_maps: Optional[list[dict[int, int]]] = None
) -> list[SMDReprOutput]:
    ref_sample_idxs = set(ref_sample_map.keys())
    n_ref_samples = len(ref_sample_map)

    if alt_group_maps is None:
        n_alt_groups = 1
    else:
        n_alt_groups = max(max(d.values()) for d in alt_group_maps) + 1

    # ref.
    with tqdm(total=n_ref_samples, desc="Processing references...", leave=False) as pbar:
        with h5py.File(ref_acts_h5_path, "r") as h5f:
            chunks = h5_chunk_sorter(list(h5f.keys()))
            n_features = h5f.attrs["num_features"]

            ref_max = np.zeros((n_ref_samples, n_features), dtype=np.float32)
            ref_max_sq = np.zeros((n_ref_samples, n_features), dtype=np.float32)

            curr_smi = 0
            for cn in chunks:
                g = h5f[cn]
                molptr = g["molptr"][:]
                indptr = g["indptr"][:]
                data = g["data"][:]

                curr_indptr = 0
                curr_data = 0
                for _ in molptr:
                    end_indptr = curr_indptr + n_features + 1
                    mol_indptr = indptr[curr_indptr:end_indptr]
                    end_data = curr_data + mol_indptr[-1]

                    if curr_smi in ref_sample_idxs:
                        nz_mask = np.diff(mol_indptr) > 0
                        ref_smi_ptr = ref_sample_map[curr_smi]

                        fast_eval_repr_smd(
                            ref_max, ref_max_sq,
                            ref_smi_ptr,
                            mol_indptr, data[curr_data:end_data],
                            nz_mask,
                        )
                        pbar.update()

                    curr_indptr = end_indptr
                    curr_data   = end_data
                    curr_smi   += 1

                # early stop
                if pbar.n == pbar.total:
                    break

    # alt.
    alt_max = np.zeros((n_ref_samples, n_alt_groups, n_features), dtype=np.float32)
    alt_max_sq = np.zeros((n_ref_samples, n_alt_groups, n_features), dtype=np.float32)

    with h5py.File(alt_acts_h5_path, "r") as h5f:
        chunks = h5_chunk_sorter(list(h5f.keys()))
        n_features = h5f.attrs["num_features"]
        n_samples = h5f.attrs["num_samples"]

        # track ref.
        curr_ref = 0
        curr_end = alt_end_map[curr_ref]
        count_alt = 0
        count_group = np.zeros(n_alt_groups, dtype=np.uint32)
        group_map = alt_group_maps[curr_ref] if alt_group_maps is not None else None

        curr_smi = 0
        for cn in chunks:
            g = h5f[cn]
            molptr = g["molptr"][:]
            indptr = g["indptr"][:]
            data   = g["data"][:]

            curr_indptr = curr_data = 0
            for _ in molptr:
                end_indptr = curr_indptr + n_features + 1
                mol_indptr = indptr[curr_indptr:end_indptr]
                end_data   = curr_data + mol_indptr[-1]

                grp = group_map[count_alt] if group_map is not None else 0
                nz_mask = np.diff(mol_indptr) > 0

                fast_eval_repr_smd(
                    alt_max[:, grp, :], alt_max_sq[:, grp, :],
                    curr_ref,
                    mol_indptr, data[curr_data:end_data],
                    nz_mask
                )

                curr_indptr = end_indptr
                curr_data = end_data
                curr_smi += 1

                count_alt += 1
                count_group[grp] += 1

                if curr_smi == curr_end:
                    alt_max[curr_ref] /= count_group[:, None]
                    alt_max_sq[curr_ref] /= count_group[:, None]

                    if curr_smi < n_samples:
                        curr_ref += 1
                        curr_end = alt_end_map[curr_ref]
                        count_alt = 0
                        count_group.fill(0)
                        group_map = (
                            alt_group_maps[curr_ref] if alt_group_maps is not None
                            else None
                        )

    ref_max_avg = ref_max.mean(axis=0)
    ref_max_var = ref_max_sq.mean(axis=0) - ref_max_avg ** 2

    alt_max_avg = np.nanmean(alt_max, axis=0)
    alt_max_var = np.nanmean(alt_max_sq, axis=0) - alt_max_avg ** 2

    std_max = np.sqrt((ref_max_var + alt_max_var) / 2)
    smd_max = (ref_max_avg - alt_max_avg) / std_max

    ref_max_hi = (ref_max[:, None, :] > alt_max).sum(axis=0)
    alt_max_hi = (alt_max > ref_max[:, None, :]).sum(axis=0)

    outs = []
    for g in range(n_alt_groups):
        for f in range(n_features):
            outs.append(
                SMDReprOutput(
                    feature=f,
                    group=str(g),
                    smd=smd_max[g, f],
                    refHi=ref_max_hi[g, f],
                    altHi=alt_max_hi[g, f]
                )
            )
    return outs
