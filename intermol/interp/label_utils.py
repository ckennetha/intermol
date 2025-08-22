import re

from rdkit import Chem
from typing import Optional

_RX_BOND = re.compile(r'[\-=#\\\/]')
_RX_BOND_DT = re.compile(r'[=#]')
_RX_BOND_DT_BRANCH = re.compile(r'[=#](\%[0-9]{2}|[0-9])')
_RX_BRANCH = re.compile(r'(\%[0-9]{2}|[0-9])')
_RX_RING = re.compile(r'[\(\)]')
_RX_NONATOM = re.compile(r'(\(|\)|\.|\%[0-9]{2}|[0-9])')
_RX_NONATOM_NOR = re.compile(r'(\)|\.|\%[0-9]{2}|[0-9])')


# get atom-to-token index mapping
def map_atom_token_idx(
    tokens: list, pattern: str=r'[a-zA-Z]'
) -> dict[int, int]:
    mapping = {}
    counter = 0
    for i, tok in enumerate(tokens):
        if re.search(pattern, tok):
            mapping[counter] = i
            counter += 1
    return mapping

# group sequential atom indices
def group_seq_atom_idx(match_atom_idx: list) -> list[list]:
    sort_mai = sorted(match_atom_idx)

    groups = []
    curr = [sort_mai[0]]
    for i in range(1, len(sort_mai)):
        if sort_mai[i] == (sort_mai[i-1] + 1):
            curr.append(sort_mai[i])
        else:
            groups.append(curr)
            curr = [sort_mai[i]]
    
    groups.append(curr)
    return groups

# check ring formed by matching atoms
def check_ring(mol: Chem.Mol, match_atom_idx: list) -> bool:
    ss = Chem.MolFromSmarts(Chem.MolFragmentToSmiles(mol, match_atom_idx))
    ss.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(ss)

    if ss.GetRingInfo().NumRings() > 0:
        return True
    else:
        return False

# list all rings with their indices
def list_rings(
    mol: Chem.Mol, atom_token_idx_map: dict, tokens: list
) -> list[list[int, set]]:
    ring_atoms = mol.GetRingInfo().AtomRings()
    rings = []
    
    for ra in ring_atoms:
        min_ra_idx = atom_token_idx_map[min(ra)]

        indices = set()
        for tk in tokens[min_ra_idx+1:]:
            if _RX_BRANCH.fullmatch(tk):
                indices.add(tk)
            elif _RX_BOND.fullmatch(tk):
                continue
            else:
                break
        
        ptr = 0
        while len(indices) > 1:
            max_indices = set()
            max_ra_idx = atom_token_idx_map[sorted(ra, reverse=True)[ptr]]
            
            for tk in tokens[max_ra_idx+1:]:
                if _RX_BRANCH.fullmatch(tk):
                    max_indices.add(tk)
                elif _RX_BOND.fullmatch(tk):
                    continue
                else:
                    break
            
            indices = indices.intersection(max_indices)
            ptr += 1
        
        rings.append([*list(indices), set(ra)])
    return rings

# check branch formed by matching atoms
def check_branch(
    mol: Chem.Mol, obj: str="molecule", match_atom_idx: Optional[list]=None
) -> bool:
    if obj == "molecule":
        atoms = [mol.GetAtomWithIdx(mai) for mai in match_atom_idx]
        if any([True for a in atoms if a.GetDegree() > 2]):
            return True
    elif obj == "substructure":
        if any([a for a in mol.GetAtoms() if a.GetDegree() > 1]):
            return True
    else:
        raise ValueError(f'{obj} is not allowed. Allowed objects: molecule and substructure.')

# list all branches with their depths
def list_branches(tokens: list) -> list[list[int, set]]:
    branches = [[0, len(tokens)-1, 0, set()]]
    stacks = []
    curr_ti = []
    depth = 0
    for t_i, t_v in enumerate(tokens):
        if t_v == '(':
            stacks.append(t_i)
            curr_ti.append(set())
            depth += 1
        elif t_v ==')':
            if stacks:
                branches.append([stacks.pop(), t_i, depth, curr_ti.pop()])
                depth -= 1
        else:
            if depth == 0:
                branches[0][-1].add(t_i)
            else:
                curr_ti[-1].add(t_i)
    return branches

# token-wise mapping
def map_token(smi: str, tokens: list, pattern: Chem.Mol) -> dict[str, list | tuple]:
    # setups
    mol = Chem.MolFromSmiles(smi)
    ats_map = map_atom_token_idx(tokens)
    
    rings = list_rings(mol, ats_map, tokens)
    branches = list_branches(tokens)

    # get matching atom indices
    match_atom_idx_s = mol.GetSubstructMatches(pattern)
    
    # iterate per matching atom indices group
    match_tokens = []
    for i_mai, match_atom_idx in enumerate(match_atom_idx_s):
        match_token_idx_s = {}
        for a_i in match_atom_idx:
            pos = ats_map[a_i]
            match_token_idx_s[a_i] = pos
        match_tokens.append(list(match_token_idx_s.values()))

        # label in-between
        ext_match_token_idx_s = []
        if len(match_atom_idx) > 1:
            match_atom_idx_g = group_seq_atom_idx(match_atom_idx)
            match_token_idx_s = [[v for k, v in match_token_idx_s.items() if k in mai] for mai in match_atom_idx_g]
            
            # add check for two-only atoms/tokens
            if ((len(match_token_idx_s) == 2) and
                (all(len(mti) == 1 for mti in match_token_idx_s))):
                tokens_end_aft = ''.join(tokens[match_token_idx_s[1][0]+1:match_token_idx_s[1][0]+3])

                is_single_or_aromatic = all([(b.GetBondType() == Chem.BondType.SINGLE or
                                              b.GetBondType() == Chem.BondType.AROMATIC)
                                              for b in pattern.GetBonds()])
                if not is_single_or_aromatic:
                    if _RX_BOND_DT.fullmatch(tokens[match_token_idx_s[1][0]-1]):
                        match_tokens[i_mai].append(
                            match_token_idx_s[1][0]-1
                            )
                        continue
                    elif _RX_BOND_DT_BRANCH.fullmatch(tokens_end_aft):
                        match_tokens[i_mai].extend([
                            match_token_idx_s[1][0]+1,
                            match_token_idx_s[1][0]+2
                            ])
                        continue
            
            # check for ring(s) and/or branch(es)
            is_ring = check_ring(mol, match_atom_idx)
            is_branch = check_branch(mol, match_atom_idx=match_atom_idx)

            num_stacks = {}
            branch_pairs = []

            for i_mti, match_token_idx in enumerate(match_token_idx_s):
                start_aft = min(match_token_idx)
                end = max(match_token_idx)

                toks_map = {}

                # include bondtype token if the token is on the beginning of open parenthesis (branch) and
                # consist of either double or triple bond(s)
                if (re.fullmatch(r'\([=#]', ''.join(tokens[start_aft-2:start_aft])) and
                    any([(b.GetBondType() == Chem.BondType.DOUBLE or
                          b.GetBondType() == Chem.BondType.TRIPLE)
                        for b in pattern.GetBonds()]) and
                    i_mti > 0):
                    for branch in branches:
                        if (start_aft-2) == branch[0]:
                            depth = branch[2]

                    if (depth-1) == 0:
                        if match_tokens[i_mai][0] in branches[0][3]:
                            toks_map[start_aft-1] = tokens[start_aft-1]
                    else:
                        bti_to_check = [bti for branch in branches
                                        if ((branch[0] < (start_aft-2)) and (branch[2] == (depth-1)))
                                        for bti in branch[3]]
                        if match_tokens[i_mai][0] in bti_to_check:
                            toks_map[start_aft-1] = tokens[start_aft-1]

                for k in range(start_aft+1, end):
                    # excl. double bond + ring closure mis-interp
                    if not ((tokens[k] == '=') and
                            _RX_BRANCH.fullmatch(tokens[k+1])):
                        toks_map[k] = tokens[k]

                # extend token(s) to include all nearest integers and single bonds, if possible
                for i in range(end+1, len(tokens)):
                    tk_i = tokens[i]
                    if _RX_NONATOM_NOR.fullmatch(tk_i):
                        toks_map[i] = tokens[i]
                    else:
                        break

                for t_k, t_v in toks_map.items():
                    if not _RX_NONATOM.fullmatch(t_v):
                        ext_match_token_idx_s.append(t_k)
                    else:
                        if _RX_RING.fullmatch(t_v) and is_branch:
                            for branch in branches:
                                if (branch[0] == t_k) or (branch[1] == t_k):
                                    branch_pairs.append([branch[0], branch[1]])
                        elif _RX_BRANCH.fullmatch(t_v):
                            if t_v not in num_stacks.keys():
                                num_stacks[t_v] = [t_k]
                            else:
                                num_stacks[t_v].append(t_k)
                        else:
                            continue
                    
            # validate num_stacks
            valid_num_stacks = []
            if is_ring:
                for ridx, idxs in num_stacks.items():
                    for ring in rings:
                        if ridx == ring[0]:
                            mai_to_check = set(match_atom_idx)
                            if (((ring[1].issuperset(mai_to_check)) or (ring[1].issubset(mai_to_check))) and
                                (len(idxs) > 1)):
                                if len(idxs) % 2 == 0:
                                    valid_num_stacks.extend(idxs)
                                else:
                                    valid_num_stacks.extend(idxs[:-1])
            
            min_mt = min(match_tokens[i_mai])
            max_mt = max(match_tokens[i_mai])
            if check_branch(pattern, obj='substructure'):
                branch_pairs = [[bs_i, be_i] for bs_i, be_i in branch_pairs if not
                                (bs_i < min_mt or be_i > max_mt)]
            else:
                branch_pairs = [[bs_i, be_i] for bs_i, be_i in branch_pairs if
                                (bs_i > min_mt or be_i < max_mt)]

            for i in sum(branch_pairs + [valid_num_stacks,], []):
                ext_match_token_idx_s.append(i)
            
            # remove one- or zero-sided single bonds from labels
            if '-' in smi:
                lookup = match_tokens[i_mai] + ext_match_token_idx_s
                singles = [t_i for t_i, t in enumerate(tokens) if t == '-']
                for t_i in singles:
                    if not (((t_i+1) in lookup) and ((t_i-1) in lookup)):
                        try:
                            ext_match_token_idx_s.remove(t_i)
                        except:
                            pass

            match_tokens[i_mai].extend(ext_match_token_idx_s)
            match_tokens[i_mai] = list(set(match_tokens[i_mai]))

    return {
        'match_atom_idx': match_atom_idx_s,
        'match_token_idx': match_tokens
    }