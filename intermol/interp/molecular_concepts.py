import re

from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem.FilterCatalog import *

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
def generate_atom_in_substructure(
    smi: str,
    tokens: list[str],
    atom_idx_to_token_idx_map: dict[int, int]
) -> list[str]:
    mol = Chem.MolFromSmiles(smi)

    out = []
    for ra_idx, rt_idx in atom_idx_to_token_idx_map.items():
        bn_idxs = Chem.FindAtomEnvironmentOfRadiusN(mol, 1, ra_idx)
        nbs = []
        for bn_idx in bn_idxs:
            bn = mol.GetBondWithIdx(bn_idx)
            bn_type = str(bn.GetBondType())

            s, e = bn.GetBeginAtomIdx(), bn.GetEndAtomIdx()
            nb_idx = s if e == ra_idx else e
            nbs.append(
                f"{BOND_SYMBOL[bn_type]}{tokens[atom_idx_to_token_idx_map[nb_idx]]}"
            )

        nbs = sorted(nbs)
        nbs_str = ''.join(
            f"({nb})" if (i+1) < len(nbs) else nb
            for i, nb in enumerate(nbs)
        )

        sma = rt_idx + nbs_str
        out.append(sma)
        if len(nbs) > 0:
            try:
                submol = Chem.MolFromSmarts(sma)
                for sa in submol.GetAtoms():
                    out.append(f"[$({Chem.MolToSmarts(submol, sa.GetIdx())})]")
            except Exception:
                out.append(f"[$({sma})]")
                for i_nb, nb in enumerate(nbs):
                    nb_bn = nb[0]
                    if nb_bn in {"=", "#", ":"}:
                        nb_at = nb[1:]
                    else:
                        nb_bn = ""
                        nb_at = nb

                    nb_nb = nbs[i_nb+1:] + nbs[:i_nb]
                    nb_nb_str = ''.join(
                        f"({nbx})" if (i_nbx + 1) < len(nb_nb) else nbx
                        for i_nbx, nbx in enumerate(nb_nb)
                    )
                    out.append(f"[$({nb_at + nb_bn + rt_idx + nb_nb_str})]")

    return out

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
            descs = self.sma_map.keys()

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

## list all branch points in the SMILES
def list_branch(tokens: list[str]) -> list:
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
        "pair_token_idxs": (0, len(tokens)-1),
        "atom_token_idx_in_branch": main_tk_at_i,
        "token_idx_in_branch": main_tk_i
    }
    return [main, *brs]
