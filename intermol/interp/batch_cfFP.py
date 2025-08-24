import argparse
from .concept_fp import ConceptFromFingerprint

def main():
    parser = argparse.ArgumentParser(
        description='''
            Identify associations between abstract molecular concepts (fragmented Smiles or Smarts strings)
            and features, given a set of molecules with their respective high-activation token(s). Default
            mode outputs all identified fragmented molecular strings. Currently, only atom-based matching by
            Morgan Fingerprint (FP) generator from RDKit is supported.
        '''
    )
    parser.add_argument('--dataset', type=str, required=True, help='Path to SMILES data per feature. Supported ext.: .msgpack.')
    parser.add_argument('--fptype', type=str, default="morgan", choices=["morgan", "rdkit", "morgan_rdkit"],
                        help='FP type used in calculation. Default: morgan.')
    parser.add_argument('--fpsize', type=int, default=2048, help='Size of the generated Morgan FP. Default: 2048.')
    parser.add_argument('--radius', type=int, default=2, help='Radius for Morgan FP calculation. Default: 2.')
    parser.add_argument('--minpath', type=int, default=1, help='Minimum number of traversed paths/bonds for RDKit FP calculation. Default: 1.')
    parser.add_argument('--maxpath', type=int, default=4, help='Maximum number of traversed paths/bonds for RDKit FP calculation. Default: 4.')
    parser.add_argument('--count_simulation', action="store_true", help="If set, use count simulation while generating the FP.")
    parser.add_argument('--chirality', action="store_true", help="If set, chirality information will be added to the generated Morgan FP.")
    parser.add_argument('--output_type', type=str, default="smiles", choices=["smiles", "smarts"],
                        help='Format of molecular motif output. Default: smiles.')
    parser.add_argument('--top_frequent_bits', type=int, default=1, help='Outputs the top-K of most common molecular FP bits per feature. '
                        'Default: 1.')
    parser.add_argument('--output_drawing', action="store_true", help='If set, outputs PNG images of every molecule highlighting FP results. '
                        'Warning: may generate many files and consume disk space.')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers to perform feature-wise FP calculations in parallel.')
    parser.add_argument('--out_prefix', type=str, default=None, help='Output file prefix. Default: None.')
    parser.add_argument('--outdir', type=str, help='Output directory. Default: current directory.')

    args = parser.parse_args()


    # init FP generator
    cff = ConceptFromFingerprint(
        fptype=args.fptype,
        fpsize=args.fpsize,
        radius=args.radius,
        minpath=args.minpath,
        maxpath=args.maxpath,
        countSimulation=args.count_simulation,
        includeChirality=args.chirality
    )
    
    cff.run_batch(
        dataset_pth=args.dataset,
        output_type=args.output_type,
        output_drawing=args.output_drawing,
        top_frequent_bits=args.top_frequent_bits,
        num_workers=args.num_workers,
        outdir_pth=args.outdir,
        out_prefix=args.out_prefix
    )


if __name__ == '__main__':
    main()
