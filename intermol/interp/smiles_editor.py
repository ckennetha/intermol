import re
import time
import random

from typing import Optional, Literal
from rdkit import Chem

from intermol.interp.molecular_concepts import _RX_TOKEN, _RX_RING, list_branch

# utils
## replace each token in a SMILES with `[MASK]`
def enum_mask_smi(
    smi: str,
    start: int = 0,
    end: Optional[int] = None,
    exclude_token: Optional[re.Pattern] = None
) -> list[tuple[str, int, str]]:
    masked = []
    tks = _RX_TOKEN.findall(smi)[start:end]
    for tk_i, tk in enumerate(tks, start=start):
        if exclude_token is not None and exclude_token.fullmatch(tk):
            continue
        m_smi = ''.join(tks[:tk_i]) + "[MASK]" + ''.join(tks[tk_i + 1:])
        masked.append((m_smi, tk_i, tk))
    return masked


class SmilesEditor:
    def __init__(self, seed: int = None):
        self.seed = seed if seed is not None else int(time.time())

        # set seed
        random.seed(seed)

        # defaults
        self.error_map = {
            "rings": self._error_rings,
            "parentheses": self._error_parentheses,
            "aromaticity": self._error_aromaticity,
            "syntax": self._error_syntax,
            "valence": self._error_valence
        }

    def _error_rings(self, tokens: list[str]) -> tuple[float, list[str]]:
        ris = list({tk for tk in tokens if _RX_RING.fullmatch(tk)})
        s = random.random()
        if s < 0.8 and ris:
            ri = random.choice(ris)
            ri_idxs = [tk_i for tk_i, tk in enumerate(tokens) if tk == ri]
            if s < 0.6:
                # remove ring index
                ri_idx = random.choice(ri_idxs)
                tokens.pop(ri_idx)

                in_s = random.random()
                if in_s < 0.25:
                    # replace with +1 or -1
                    ri = int(ri.lstrip("%"))
                    if ri == 1:
                        tokens.insert(ri_idx, "2")
                    else:
                        ri = ri + random.choice((-1, 1))
                        tokens.insert(ri_idx, str(ri) if ri < 10 else f"%{ri}")
                elif in_s < 0.5:
                    # replace with existing ring index
                    ri_p1 = len(ris) + 1 # (+1)
                    ris.append(str(ri_p1) if ri_p1 < 10 else f"%{ri}")
                    ris.remove(ri)
                    tokens.insert(ri_idx, random.choice(ris))
            else:
                # duplicate ring opening index
                ri_idx_i = random.choice(range(0, len(ri_idxs), 2))
                tokens.insert(ri_idxs[ri_idx_i], ri)
        else:
            # add ring index
            ri_p1 = len(ris) + 1 # (+1)
            ris.append(str(ri_p1) if ri_p1 < 10 else f"%{ri}")
            tokens[random.randrange(0, len(tokens))] = random.choice(ris)
        return s, tokens

    def _error_parentheses(self, tokens: list[str]) -> tuple[float, list[str]]:
        s = random.random()
        if s < 0.2:
            # add parenthesis
            tokens.insert(
                random.randrange(1, len(tokens) + 1),
                random.choice(('(', ')'))
            )
        elif s < 0.4:
            # remove parenthesis
            p_idxs = [tk_i for tk_i, tk in enumerate(tokens) if tk == '(' or tk == ')']
            if p_idxs:
                tokens.pop(random.choice(p_idxs))
        elif s < 0.6:
            # switch parenthesis
            op_idxs = [tk_i for tk_i, tk in enumerate(tokens) if tk == '(']
            cp_idxs = [tk_i for tk_i, tk in enumerate(tokens) if tk == ')']
            if op_idxs and cp_idxs:
                tokens[random.choice(op_idxs)] = ")"
                tokens[random.choice(cp_idxs)] = "("
        elif s < 0.8:
            # '(' > ')'
            op_idxs = [tk_i for tk_i, tk in enumerate(tokens) if tk == '(']
            if op_idxs:
                tokens[random.choice(op_idxs)] = ")"
        else:
            # ')' > '('
            cp_idxs = [tk_i for tk_i, tk in enumerate(tokens) if tk == ')']
            if cp_idxs:
                tokens[random.choice(cp_idxs)] = "("
        return s, tokens

    def _error_aromaticity(self, tokens: list[str]) -> tuple[float, list[str]]:
        s = random.random()
        if s < 0.2:
            # non-arom to arom token
            naa_idxs = [
                tk_i for tk_i, tk in enumerate(tokens)
                if tk in {"C", "N", "O", "S", "P", "B"}
            ]
            if naa_idxs:
                naa_idx = random.choice(naa_idxs)
                tokens[naa_idx] = tokens[naa_idx].lower()
        elif s < 0.4:
            # convert arom with 1 pi electron
            aa1_idxs = [
                tk_i for tk_i, tk in enumerate(tokens)
                if tk == "c" or tk == "n"
            ]
            if aa1_idxs:
                aa1_idx = random.choice(aa1_idxs)
                tokens.pop(aa1_idx)
                tokens[aa1_idx:aa1_idx] = random.choice([
                    "[nH]",
                    ["n", "(", "C", ")"],
                    "o",
                    "s",
                    ["c", "(", "=", "O", ")"],
                    ["c", "(", "=", "N", ")"],
                    "C",
                    "N",
                    ["c", "c"],
                    ["c", "n"],
                    ["n", "c"],
                    ["n", "n"]
                ])
        elif s < 0.6:
            # insert random 'c' or 'n' into arom ring
            aa_idxs = [tk_i for tk_i, tk in enumerate(tokens) if tk.islower()]
            if aa_idxs:
                tokens.insert(
                    random.choice(aa_idxs) + random.choice((0, 1)),
                    random.choice(("c", "n"))
                )
        elif s < 0.8:
            # convert arom with 2 pi electron
            aa2_idxs = [
                tk_i for tk_i, tk in enumerate(tokens)
                if tk in {"[nH]", "o", "s"}
            ]
            if aa2_idxs:
                tokens[random.choice(aa2_idxs)] = random.choice(("c", "n"))
        else:
            # remove arom with 1 pi electron
            aa1_idxs = [
                tk_i for tk_i, tk in enumerate(tokens)
                if tk == "c" or tk == "n"
            ]
            if aa1_idxs:
                tokens.pop(random.choice(aa1_idxs))
        return s, tokens

    def _error_syntax(self, tokens: list[str]) -> tuple[float, list[str]]:
        s = random.random()
        bp_tks = ['-', '=', '#', '(', ')']
        if s < 0.1:
            # mutate at idx = 0
            tokens.insert(0, random.choice(bp_tks))
        elif s < 0.2:
            # mutate at idx = -1
            tokens.insert(len(tokens), random.choice(bp_tks))
        elif s < 0.3:
            # duplicate bond token
            bn_tks = set(bp_tks[:3])
            bn_idxs = [tk_i for tk_i, tk in enumerate(tokens) if tk in bn_tks]
            if bn_idxs:
                tokens.insert(
                    random.choice(bn_idxs) + random.randint(0, 1),
                    random.choice(bn_tks)
                )
        elif s < 0.4:
            # add bond token before ring opening
            or_idxs = []
            ori = set()
            for i, tk in enumerate(tokens):
                if _RX_RING.fullmatch(tk):
                    if tk not in ori:
                        ori.add(tk)
                        or_idxs.append(i)
                    else:
                        ori.remove(tk)
            if or_idxs:
                tokens.insert(random.choice(or_idxs), random.choice(bp_tks[:3]))
        elif s < 0.6:
            # add bond token before branch opening
            # or duplicate branch opening
            op_idxs = [i for i, tk in enumerate(tokens) if tk == "("]
            if op_idxs:
                tokens.insert(
                    random.choice(op_idxs),
                    random.choice(bn_tks[:3]) if s < 0.5 else "("
                )
        elif s < 0.8:
            try:
                ri_idx = tokens.index("1")
                p_idxs = [
                    tk_i for tk_i, tk in enumerate(tokens[:ri_idx])
                    if tk == "(" or tk == ")"
                ]

                near_op = None
                if p_idxs:
                    near_p = p_idxs.pop()
                    if tokens[near_p] == "(":
                        near_op = near_p
                in_s = random.random()
            except ValueError:
                ri_idx, near_op = None, None

            if ri_idx:
                if (in_s < 0.5) or (near_op is None):
                    # truncate tokens before first ring opening
                    tr_tks = tokens[ri_idx:]
                else:
                    # between branch opening and first ring opening
                    tr_tks = tokens[:near_op+1] + tokens[ri_idx:]
                # max 25% of tokens retained
                if (len(tr_tks) / len(tokens)) >= 0.25:
                    tokens = tr_tks
        else:
            # empty branch
            brs = list_branch(tokens)
            if len(brs) > 1:
                br_ps = [
                    (br["pair_token_idxs"][0], br["pair_token_idxs"][1])
                    for br in brs[1:]
                ]
                otk, ctk = random.choice(br_ps)
                tokens = tokens[:otk+1] + tokens[ctk:]
                in_s = random.random()
                if in_s < 0.5:
                    # insert random non-atom or non-branch tokens
                    ris = [tk for tk in tokens if _RX_RING.fullmatch(tk)]
                    tokens.insert(otk+1, random.choice(bp_tks[:3] + ris))
        return s, tokens

    def _error_valence(self, smi: str, n: int = 1) -> str:
        # increase bond order
        mol = Chem.MolFromSmiles(smi)
        for _ in range(n):
            ms = mol.GetSubstructMatches(Chem.MolFromSmarts("[A;!h]-,=*"))
            if ms:
                at_idxs = random.choice(ms)
                bn = mol.GetBondBetweenAtoms(*at_idxs)
                if bn.GetBondType() == Chem.rdchem.BondType.SINGLE:
                    bn.SetBondType(
                        random.choice((
                            Chem.rdchem.BondType.DOUBLE,
                            Chem.rdchem.BondType.TRIPLE
                        ))
                    )
                else:
                    bn.SetBondType(Chem.rdchem.BondType.TRIPLE)

                try:
                    smi = Chem.MolToSmiles(mol)
                except Exception:
                    pass
        return smi

    def generate_invalid(
        self,
        smi: str, n: int = 1,
        error: Literal[
            'all',
            'rings',
            'parentheses',
            'aromaticity',
            'syntax',
            'valence'
        ] = 'all'
    ) -> list[dict[str, str | int]]:
        if error == "all":
            error = list(self.error_map.keys())
        else:
            if error not in self.error_map:
                raise ValueError(
                    "`error` must be one of "
                    f"{['all'] + list(self.error_map.keys())}, "
                    f"got {error!r}"
                )

        outs = []
        tokens = _RX_TOKEN.findall(smi)
        for et in error:
            if et == "valence":
                e_smi = self._error_valence(smi, n)
            else:
                e_tks = tokens.copy()
                for _ in range(n):
                    e_tks = self.error_map[et](e_tks)
                e_smi = ''.join(e_tks)
            outs.append({"error": et, "error_Smiles": e_smi})
        return outs

    @staticmethod
    def generate_noncanon(smi: str, n: int = 5, reps: int = 10) -> list[str]:
        if reps < n:
            raise ValueError(
                "`reps` must be larger than `n`, ",
                f"got n={n} and reps={reps}."
            )

        mol = Chem.MolFromSmiles(smi)
        c_smi = Chem.MolToSmiles(mol, canonical=True)
        nc_smiles = set()
        for _ in range(reps):
            nc_smi = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
            if nc_smi != c_smi and nc_smi not in nc_smiles:
                nc_smiles.add(nc_smi)
                if len(nc_smiles) == n:
                    break
        return list(nc_smiles)
