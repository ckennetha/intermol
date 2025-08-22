import os
import json
import random

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from msgpack import Unpacker, Packer
from pathlib import Path
from PIL.PngImagePlugin import PngImageFile
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw
from tqdm import tqdm
from typing import Optional, Union, Callable

RDLogger.DisableLog('rdApp.*')

def to_molecule(smi: str) -> Chem.Mol:
    return Chem.MolFromSmiles(smi)

def calculate(
    fpgen: Callable, fpgen_type: str, mol: Chem.Mol,
    fromAtoms: list[int]=[], ignoreAtoms: list[int]=[]
) -> dict[int, tuple[tuple[int, int]]]:
    ao = AllChem.AdditionalOutput()
    
    if fpgen_type == "morgan":
        ao.CollectBitInfoMap()
        _ = fpgen.GetFingerprint(
            mol, fromAtoms=fromAtoms, ignoreAtoms=ignoreAtoms, additionalOutput=ao
        )
        return ao.GetBitInfoMap()
    else:
        ao.CollectBitPaths()
        _ = fpgen.GetFingerprint(
            mol, fromAtoms=fromAtoms, ignoreAtoms=ignoreAtoms, additionalOutput=ao
        )
        return ao.GetBitPaths()

def draw_morgan(mol: Chem.Mol, bi: dict[int, tuple[tuple[int, int]]]) -> PngImageFile:
    coll_bits = [(mol, i, bi) for i in bi]
    labels = [str(i[1]) for i in coll_bits]
    return Draw.DrawMorganBits(
        coll_bits, molsPerRow=min(len(coll_bits), 5), legends=labels
    )

def get_fragment(
    fptype: str, mol: Chem.Mol, output_type: str='smiles',
    rootedAtAtoms: Optional[list[int]]=None, radius: Optional[int]=None,
    bondsInPaths: Optional[list[int]]=None
) -> str:
    fragmentor = Chem.MolFragmentToSmiles if output_type == 'smiles' else Chem.MolFragmentToSmarts

    if fptype == 'morgan':
        if radius == 0:
            return fragmentor(mol, atomsToUse=rootedAtAtoms)

        root_atom = rootedAtAtoms[0]
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, root_atom)
        atoms = set()
        for bond_idx in env:
            bond = mol.GetBondWithIdx(bond_idx)
            atoms.add(bond.GetBeginAtomIdx())
            atoms.add(bond.GetEndAtomIdx())

        if not atoms:
            return fragmentor(mol, atomsToUse=rootedAtAtoms)

        if output_type == 'smiles':
            return fragmentor(mol, atomsToUse=list(atoms), bondsToUse=env, rootedAtAtom=root_atom)
        else:
            return fragmentor(mol, atomsToUse=list(atoms), bondsToUse=env)
    else:
        atoms = set()
        for bond_idx in bondsInPaths:
            bond = mol.GetBondWithIdx(bond_idx)
            atoms.add(bond.GetBeginAtomIdx())
            atoms.add(bond.GetEndAtomIdx())

        return fragmentor(mol, atomsToUse=list(atoms), bondsToUse=list(bondsInPaths))

def init_fpgen(
    fptype: str='morgan_rdkit', fpsize: int=2048, radius: int=2, minPath:int=1, maxpath: int=4,
    countSimulation: bool=False, includeChirality: bool=False
) -> None:
    global FPGEN
    FPGEN = dict()

    if fptype in ['morgan', 'morgan_rdkit']:
        # init Morgan FP
        FPGEN["morgan"] = AllChem.GetMorganGenerator(
            radius=radius, fpSize=fpsize,
            countSimulation=countSimulation,
            includeChirality=includeChirality
        )
    
    if fptype in ['rdkit', 'morgan_rdkit']:
        # init RDKit FP
        FPGEN["rdkit"] = AllChem.GetRDKitFPGenerator(
            minPath=minPath, maxPath=maxpath, fpSize=fpsize,
            countSimulation=countSimulation
        )

def wrap_single(
    fptype: str, mol: Chem.Mol, fromAtoms: list[int]=[], ignoreAtoms: list[int]=[],
    output_drawing: bool=False
) -> dict[str, dict[int, tuple[tuple[int, int]]]]:
    global FPGEN
    result = {}

    if fptype in ['morgan', 'morgan_rdkit']:
        bi = calculate(
            fpgen=FPGEN["morgan"], fpgen_type="morgan", mol=mol,
            fromAtoms=fromAtoms, ignoreAtoms=ignoreAtoms
        )
        result["bits_morgan"] = bi
    
    if fptype in ['rdkit', 'morgan_rdkit']:
        bp = calculate(
            fpgen=FPGEN["rdkit"], fpgen_type="rdkit", mol=mol,
            fromAtoms=fromAtoms, ignoreAtoms=ignoreAtoms
        )
        result["bits_rdkit"] = bp

    if output_drawing and (fptype in ['morgan', 'morgan_rdkit']):
        drawing_morgan = draw_morgan(mol, bi)
        result["drawing_morgan"] = drawing_morgan

    return result

def wrap_batch(
    fptype: str, smiles: list[str], fromAtoms: list[list[int]]=[], ignoreAtoms: list[list[int]]=[],
    top_frequent_bits: int=1, output_type: str='smiles', output_drawing: bool=False,
    f: Optional[str]=None
) -> dict:
    global FPGEN
    n_smi = len(smiles)
    bits_count = defaultdict(Counter)
    types_count = defaultdict(lambda: defaultdict(Counter))
    smiles_match = defaultdict(lambda: defaultdict(list))

    if not fromAtoms:
        fromAtoms = [[]] * n_smi

    if not ignoreAtoms:
        ignoreAtoms = [[]] * n_smi

    for i, smi in enumerate(smiles):
        mol = to_molecule(smi)
        result = wrap_single(
            fptype=fptype, mol=mol, fromAtoms=fromAtoms[i],
            ignoreAtoms=ignoreAtoms[i], output_drawing=output_drawing
        )
        
        if fptype in {'morgan', 'morgan_rdkit'}:
            res_morgan = result["bits_morgan"]
            for bit_k, bit_v in res_morgan.items():
                bits_count["morgan"][bit_k] += 1
                smiles_match["morgan"][bit_k].append(i)
                
                for bi_pair in bit_v:
                    at_idx, rad = bi_pair
                    fragment = get_fragment(
                        fptype, mol, output_type, rootedAtAtoms=[at_idx], radius=rad
                    )
                    types_count["morgan"][bit_k][fragment] += 1

        if fptype in {'rdkit', 'morgan_rdkit'}:
            cache = defaultdict(lambda: [None, Counter()])
            res_rdkit = result["bits_rdkit"]
            for bit_k, bit_v in res_rdkit.items():
                bits_count["rdkit"][bit_k] += 1
                smiles_match["rdkit"][bit_k].append(i)

                if cache[bit_v][0]:
                    types_count["rdkit"][bit_k].update(cache[bit_v][1])
                else:
                    cache[bit_v][0] = bit_k
                for bip in bit_v:
                    fragment = get_fragment(
                        fptype, mol, output_type, bondsInPaths=bip
                    )
                    types_count["rdkit"][bit_k][fragment] += 1
                    cache[bit_v][1][fragment] += 1

    tops = {
        fpt: bit_count.most_common(top_frequent_bits) for fpt, bit_count in bits_count.items()
    }

    outs = defaultdict()
    for fpt, bit_count in tops.items():
        out = defaultdict()
        for bc in bit_count:
            out_0 = [bc[1], n_smi, round(bc[1] / n_smi, 2)]
            out[str(bc[0])] = [out_0, smiles_match[fpt][bc[0]], dict(types_count[fptype][bc[0]])]
        outs[fpt] = out

    return f, outs


class ConceptFromFingerprint():
    def __init__(
        self, fptype: str='morgan_rdkit', fpsize: int=2048,
        radius: int=2, minpath:int=1, maxpath: int=4, **kwargs
    ):
        # sanity check
        try:
            assert fptype in ['morgan', 'rdkit', 'morgan_rdkit']
        except:
            raise ValueError(f'{type} fingerprint is currently not supported.')
        
        self.fptype = fptype
        self.fpsize = fpsize
        self.radius = radius
        self.minpath = minpath
        self.maxpath = maxpath
        
        self.countSimulation = kwargs.get("countSimulation", False)
        self.includeChirality = kwargs.get("includeChirality", False)
    
    def run_single(
        self, smi: str, fromAtoms: list[int]=[], ignoreAtoms: list[int]=[],
        output_drawing: bool=False
    ) -> dict[str, dict[int, tuple[tuple[int, int]]]]:
        init_fpgen(
            self.fptype, self.fpsize, self.radius, self.minpath, self.maxpath,
            self.countSimulation, self.includeChirality
        )
        mol = to_molecule(smi)
        
        return wrap_single(
            fptype=self.fptype, mol=mol, fromAtoms=fromAtoms,
            ignoreAtoms=ignoreAtoms, output_drawing=output_drawing
        )
    
    def run_batch(
        self, dataset_pth: str, output_type: str='smiles',
        top_frequent_bits: int=1, output_drawing: bool=False,
        outdir_pth: Optional[str]=None, out_prefix: Optional[str]=None, num_workers: int=1
    ) -> None:
        # sanity check output_type
        try:
            assert output_type in ['smiles', 'smarts']
        except:
            raise ValueError(f'{output_type} output type is not supported.')

        packer = Packer()
        out_f = open(
            os.path.join(
                outdir_pth,
                f"{f'{out_prefix}_' if out_prefix else ''}cfFP-top.msgpack"),
        'wb')

        with open(dataset_pth, 'rb') as h:
            unpacker = Unpacker(h, raw=False)
        
            for obj in unpacker:
                fs = list(obj.keys())
                pbar = tqdm(total=len(fs), desc="Processing per feature...")

                with ProcessPoolExecutor(
                    max_workers=num_workers,
                    initializer=init_fpgen,
                    initargs=(
                        self.fptype, self.fpsize, self.radius, self.minpath, self.maxpath,
                        self.countSimulation, self.includeChirality
                    )
                ) as exec:
                    p_wb = partial(
                        wrap_batch,
                        fptype=self.fptype, ignoreAtoms=[],
                        top_frequent_bits=top_frequent_bits,
                        output_type=output_type, output_drawing=output_drawing
                    )
                    futures = [exec.submit(
                        p_wb, f=f, smiles=obj[f]["smiles"], fromAtoms=obj[f]["atom_idx"]
                    ) for f in fs]
                    
                    for future in as_completed(futures):
                        out = future.result()
                        if out:
                            out_f.write(packer.pack({out[0]: out[1]}))
                            pbar.update()

        out_f.close()


class UnionFind():
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, mol_i: int) -> int:
        root = self.parent[mol_i]

        if self.parent[root] != root:
            self.parent[mol_i] = self.find(root)
            return self.parent[mol_i]
        
        return root

    def union(self, mol_i: int, mol_j: int):
        root_i = self.find(mol_i)
        root_j = self.find(mol_j)
        
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1

    def get_groups(self) -> defaultdict[list]:
        groups = defaultdict(list)
        for i, root in enumerate(self.parent):
            #root = self.find(root)
            groups[root].append(i)
        return groups


class ConceptFromFingerprintBatchAnalysis():
    def __init__(
        self,
        out_cff_top_batch_pth: Union[str, Path], out_prefix: Optional[str]=None,
        outdir_pth: Optional[Union[str, Path]]=None
    ):
        self.cff_tb = out_cff_top_batch_pth
        
        out_fn = f"{out_prefix}_cfFP-desc.json"
        if isinstance(outdir_pth, Path):
            self.out_fn = outdir_pth / out_fn
        else:
            self.out_fn = os.path.join(outdir_pth, out_fn)

    def analyze(self, threshold: float) -> None:
        res = {}
        with open(self.cff_tb, "rb") as h:
            unp = Unpacker(h, raw=False)
            for obj in unp:
                f = next(iter(obj))
                
                new_res = {}
                pbar = tqdm(total=len(obj[f]), leave=False)
                for fp, fp_top in obj[f].items():
                    pbar.set_description(f"Analyzing feature {f} ...")
                    new_fp_top = {}

                    for bits, bits_d in fp_top.items():
                        noPassThresh = True
                        pct = bits_d[0][-1]
                        
                        if pct >= threshold:
                            noPassThresh = False
                            top_patterns = self._group_patterns(bits_d[-1]) if len(bits_d[-1]) > 1 else bits_d[-1]

                            desc_bits_d = {
                                "hitRate": f"{bits_d[0][0]}/{bits_d[0][1]} ({pct:.2f})",
                                "topPatterns": top_patterns
                            }
                            new_fp_top[bits] = desc_bits_d
                            break
                        
                        if noPassThresh:
                            break

                    new_res[fp] = new_fp_top
                    pbar.update()
                res[f] = new_res

        with open(self.out_fn, "w") as h:
            json.dump(res, h)

    def _is_exact_match(self, mol1: Chem.Mol, mol2: Chem.Mol) -> bool:
        Chem.SanitizeMol(mol1), Chem.SanitizeMol(mol2)
        return mol1.HasSubstructMatch(mol2) and mol2.HasSubstructMatch(mol1)
    
    def _group_patterns(self, patterns: dict[str, int]) -> dict[str, int]:
        smarts_list = list(patterns.keys())
        mols = [Chem.MolFromSmarts(smarts) for smarts in smarts_list]
        counts = [count for count in patterns.values()]
        uf = UnionFind(size=len(mols))

        for mol_i, mol1 in enumerate(mols):
            for mol_j, mol2 in enumerate(mols[mol_i+1:]):
                if self._is_exact_match(mol1, mol2):
                    uf.union(mol_i, mol_i + 1 + mol_j)

        groups = uf.get_groups()
        collates = Counter()
        for root, sma_idxs in groups.items():
            for sma_idx in sma_idxs:
                collates[smarts_list[root]] += counts[sma_idx]
        return dict(collates)


# .msgpack
def prep_data(
    dataset_pth: str, outfile_pth: str, n_feats: int, excl_nta: bool=False,
    random_weight: Optional[float]=None, unsele_feats: Optional[set[int]]=None,
    unsele_smiles: Optional[set[str]]=None, max_sample: Optional[int]=None,
    max_patience: Optional[int]=3
) -> None:
    # sanity check
    if random_weight:
        assert 0.0 <= random_weight <= 1.0
    if not unsele_feats:
        unsele_feats = set()

    init_n_unsele = len(unsele_feats)

    packer = Packer()
    out_f = {str(f): {"smiles": [], "atom_idx": []} for f in range(n_feats)
             if str(f) not in unsele_feats}
    counter = Counter([(str(f), 0) for f in range(n_feats)
                       if str(f) not in unsele_feats])

    if max_sample:
        curr_patience = 0
        last_n_unsele = init_n_unsele

    with open(dataset_pth, 'rb') as h:
        unpacker = Unpacker(h, raw=False)
        
        for c, obj in enumerate(unpacker):
            if max_sample and (len(counter) == 0):
                break

            if random_weight:
                is_included = random.choices(
                    [True, False], weights=[random_weight, 1.0-random_weight], k=len(obj)
                )
                obj = {smi: data for (smi, data), is_incl in zip(obj.items, is_included) if is_incl}

            for smi, smi_data in tqdm(obj.items(), desc=f"Processing chunk {c}...", leave=False):
                if max_sample and (len(counter) == 0):
                    break
                
                for f, at_idxs in zip(smi_data["features"], smi_data["atom_idxs"]):
                    f = str(f)
                    if (f in unsele_feats) or ((not at_idxs) and excl_nta): # only include if at_idxs is not empty
                        continue
                    
                    if unsele_smiles and (smi in unsele_smiles[f]):
                        continue
                    
                    out_f[f]["smiles"].append(smi)
                    out_f[f]["atom_idx"].append(at_idxs)
                    counter[f] += 1

                if max_sample:
                    for f, count in list(counter.items()):
                        if count >= max_sample:
                            del counter[f]
                            unsele_feats.add(f)
            
            if max_sample:
                curr_n_unsele = len(unsele_feats)
                if curr_n_unsele == last_n_unsele:
                    curr_patience += 1
                    if curr_patience >= max_patience:
                        break
                else:
                    curr_patience = 0

                print(f"Saved features at chunk {c}: {curr_n_unsele - init_n_unsele}; "
                      f"Patience: {curr_patience}")
                last_n_unsele = curr_n_unsele

    with open(outfile_pth, "wb") as out_fn:
        out_fn.write(packer.pack(out_f))
