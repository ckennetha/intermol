import re
import bisect

from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.FilterCatalog import *
from tqdm.auto import tqdm
from typing import Optional

# pre-compiled regexes
_RX_TOKEN = re.compile(
    r'(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'
)
_RX_ATOM = re.compile(r'[a-zA-Z\*]') # loose atom check
_RX_BOND = re.compile(r'[\-=#\$\\\/]')
_RX_RING = re.compile(r'(\%[0-9]{2}|[0-9])')
_RX_BRANCH = re.compile(r'[\(\)]')

BOND_SYMBOL = {
    "SINGLE": "", "DOUBLE": "=", "TRIPLE": "#", "AROMATIC": ":"
}

# core funcs
## atom-in-substructure
class AtomInSubstructureSMARTS:
    def __init__(self, radius: int = 1):
        self.radius = radius

    def generate(
        self,
        smi: str,
        tokens: list[str],
        atom_idx_to_token_idx_map: dict[int, int],
        token_idx_sele: Optional[list[int]] = None
    ) -> list[str]:
        mol = Chem.MolFromSmiles(smi)
        if token_idx_sele is not None:
            atom_idxs = [
                ra_idx for ra_idx, rt_idx in atom_idx_to_token_idx_map.items()
                if rt_idx in token_idx_sele
            ]
        else:
            atom_idxs = list(atom_idx_to_token_idx_map.keys())

        out = []
        for ra_idx in atom_idxs:
            ra = mol.GetAtomWithIdx(ra_idx)
            out.append(ra.GetSmarts())
            for rad in range(1, self.radius + 1):
                bn_idxs = set(Chem.FindAtomEnvironmentOfRadiusN(mol, rad, ra_idx))

                # build adjacency dict
                adj = defaultdict(list)
                for bond in (mol.GetBondWithIdx(i) for i in bn_idxs):
                    s, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    sym = BOND_SYMBOL[str(bond.GetBondType())]
                    adj[s].append((sym, e))
                    adj[e].append((sym, s))

                sma = self._build_tree(
                    mol, ra_idx, -1, adj, tokens, atom_idx_to_token_idx_map
                )
                out.extend([sma, f"[$({sma})]"])

        return out

    def _build_tree(
        self,
        mol: Chem.Mol,
        root_atom_idx: int,
        parent_atom_idx: int,
        adj: dict[int, list[tuple[str, int]]],
        tokens: list[str],
        atom_idx_to_token_idx_map: dict[int, int],
        atom_visit: set = None
    ) -> str:
        if atom_visit is None:
            atom_visit = set()
        atom_visit.add(root_atom_idx)

        root_atom = mol.GetAtomWithIdx(root_atom_idx)
        root_atom_str = root_atom.GetSmarts()

        child_strs = sorted(
            sym + self._build_tree(
                mol,
                nb,
                root_atom_idx,
                adj,
                tokens,
                atom_idx_to_token_idx_map,
                atom_visit
            ) for sym, nb in adj.get(root_atom_idx, [])
            if (nb != parent_atom_idx) and (nb not in atom_visit)
        )

        nbs_str = ''.join(
            f"({ch})" if (ch_i + 1) < len(child_strs) else ch
            for ch_i, ch in enumerate(child_strs)
        )
        return root_atom_str + nbs_str


## label tokens from SMARTS
class BatchLabelFromSmarts():
    def __init__(
        self, smarts_map: dict[str | int, str],
        prefilter_smarts: bool = False
    ):
        # build SMARTS mapping & FilterCatalog
        self.sma_map = {}

        self.prefilter_smarts = False
        if prefilter_smarts:
            self.prefilter_smarts = True
            self.catalog = FilterCatalog()
            for desc, sma in smarts_map.items():
                sma = Chem.MolFromSmarts(sma)
                self.sma_map[desc] = sma
                sm = SmartsMatcher(sma)
                e = FilterCatalogEntry(desc, sm)
                self.catalog.AddEntry(e)
        else:
            self.sma_map = {
                desc: Chem.MolFromSmarts(sma)
                for desc, sma in smarts_map.items()
            }

    def _label(
        self, smiles: str, descs: list[str | int]
    ) -> dict[str, list[list[int]]]:
        mol = Chem.MolFromSmiles(smiles)
        tokens = _RX_TOKEN.findall(smiles)
        at_idx_to_tk_idx_map = map_atom_idx_to_token_idx(tokens)

        label = {}
        for desc in descs:
            sma = self.sma_map[desc]
            tk_idxs = [
                [at_idx_to_tk_idx_map[at_idx] for at_idx in at_idxs]
                for at_idxs in mol.GetSubstructMatches(sma, useChirality=True)
            ]

            if len(tk_idxs) == 0:
                continue
            label[desc] = tk_idxs
        return label

    def batch_label(
        self, smiles: list[str], n_threads: int = 1
    ) -> dict[str, dict[str, list[list[int]]]]:
        if self.prefilter_smarts:
            # filter
            fcs = RunFilterCatalog(self.catalog, smiles, n_threads)
            descs = [[e.GetDescription() for e in fc] for fc in fcs]
        else:
            descs = [list(self.sma_map.keys())] * len(smiles)

        # label
        n_smiles = len(smiles)
        labels = {}
        with tqdm(total=n_smiles, desc="Labeling SMILES...") as pbar:
            for smi, desc in zip(smiles, descs):
                labels[smi] = self._label(smi, desc)
                pbar.update()
        return labels


# utils
## map atom indices to the corresp. token indices
def map_atom_idx_to_token_idx(tokens: list[str]) -> dict[int, int]:
    mapping = {}
    ctr = 0
    for i_tk, tk in enumerate(tokens):
        if _RX_ATOM.search(tk):
            mapping[ctr] = i_tk
            ctr += 1
    return mapping

## map bond token indices to the corresp. paired atom indices
def map_bond_token_idx_to_pair_atom_idx(
    tokens: list[str], token_idx_to_atom_idx_map: dict[int, int]
) -> dict[int, int]:
    n_tokens = len(tokens)
    atom_token_pos = list(token_idx_to_atom_idx_map.keys())
    mapping = {}

    cp_ring_idxs = set()
    cp_ring_idx_atom = {}
    for tk_i, tk in enumerate(tokens):
        if _RX_RING.fullmatch(tk):
            if tk not in cp_ring_idxs:
                # within ring
                cp_ring_idxs.add(tk)
                ins = bisect.bisect_left(atom_token_pos, tk_i)
                cp_ring_idx_atom[tk] = atom_token_pos[ins - 1]
            else:
                # outside ring
                cp_ring_idxs.discard(tk)
            continue

        if _RX_BOND.fullmatch(tk):
            ins = bisect.bisect_left(atom_token_pos, tk_i)
            left_at_i = token_idx_to_atom_idx_map[atom_token_pos[ins - 1]]

            next_tk = tokens[tk_i + 1] if (tk_i + 1) < n_tokens else None
            right_at_i = (
                token_idx_to_atom_idx_map[cp_ring_idx_atom[next_tk]]
                if next_tk in cp_ring_idxs
                else token_idx_to_atom_idx_map[atom_token_pos[ins]]
            )

            mapping[tk_i] = (left_at_i, right_at_i)
    return mapping

## list all branch points in the SMILES
def list_branch(tokens: list[str]) -> list[dict]:
    main_tk_i = []
    main_tk_at_i = []
    brs = []

    stacks = []
    curr_tk_i = []
    curr_tk_at_i = []
    depth = 0
    for tk_i, tk in enumerate(tokens):
        if tk == '(':
            stacks.append(tk_i)
            curr_tk_i.append([])
            curr_tk_at_i.append([])
            depth += 1
        elif tk == ')':
            if stacks:
                brs.append({
                    "depth": depth,
                    "pair_token_idxs": (stacks.pop(), tk_i),
                    "atom_token_idx_in_branch": curr_tk_at_i.pop(),
                    "token_idx_in_branch": curr_tk_i.pop()
                })
                depth -= 1
        else:
            if depth == 0:
                main_tk_i.append(tk_i)
                if _RX_ATOM.search(tk):
                    main_tk_at_i.append(tk_i)
            else:
                curr_tk_i[-1].append(tk_i)
                if _RX_ATOM.search(tk):
                    curr_tk_at_i[-1].append(tk_i)

    main = {
        "depth": 0,
        "pair_atom_token_idxs": (0, len(tokens)-1),
        "atom_token_idx_in_branch": main_tk_at_i,
        "token_idx_in_branch": main_tk_i
    }
    return [main, *brs]

## list all rings in the SMILES
def list_ring(tokens: list[str], mol: Chem.Mol) -> list:
    ai_to_ti = map_atom_idx_to_token_idx(tokens)

    ring_atoms = mol.GetRingInfo().AtomRings()
    rings = []
    for ras in ring_atoms:
        s_ra = min(ras)
        s_ra_i = ai_to_ti[s_ra] # token at s_ra located before ring index

        s_ri_is = set()
        for tk in tokens[s_ra_i+1:]:
            if _RX_RING.fullmatch(tk):
                s_ri_is.add(tk)
            elif _RX_BOND.fullmatch(tk):
                continue
            else:
                break

        ras_rev = sorted(ras, reverse=True)
        ptr = 0
        e_ra_i = None
        within_ra_is = []
        while len(within_ra_is) == 0:
            e_ri_is = set()
            e_ra_i = ai_to_ti[ras_rev[ptr]]
            for tk in tokens[e_ra_i+1:]:
                if _RX_RING.fullmatch(tk):
                    e_ri_is.add(tk)
                elif _RX_BOND.fullmatch(tk):
                    continue
                else:
                    break
            within_ra_is.extend(s_ri_is.intersection(e_ri_is))
            ptr += 1

        rings.append({
            "index": ''.join(sorted(within_ra_is)),
            "pair_atom_token_idxs": (s_ra_i, e_ra_i),
            "atom_token_idx_in_ring": [ai_to_ti[ra] for ra in ras]
        })
    return rings

## list all branch and ring syntax pairs in the SMILES
def list_syntax_pair(
    tokens: list[str]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    bp, rp = [], []
    b_stack, r_map = [], dict()
    for tk_i, tk in enumerate(tokens):
        if tk == "(":
            b_stack.append(tk_i)
        elif tk == ")":
            bp.append((b_stack.pop(), tk_i))
        elif _RX_RING.fullmatch(tk):
            if tk in r_map:
                rp.append((r_map.pop(tk), tk_i))
            else:
                r_map[tk] = tk_i
    return bp, rp
