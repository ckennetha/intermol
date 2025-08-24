import argparse

from pathlib import Path
from .concept_fp import ConceptFromFingerprintBatchAnalysis

def main():
    parser = argparse.ArgumentParser(
        description='''
            Analyze 'top' output from ConceptFromFingerprint by identifying high-occurrence bits based on the given
            threshold(s). The output includes the hit rate and top grouped patterns for each feature-bit pair in
            JSON format.
        '''
    )
    parser.add_argument('--outdir_cff_top_batch', type=str, required=True, help='Path to the directory containing ConceptFromFingerprint '
                        'batch processing output file.')
    parser.add_argument('--threshold', type=float, default=0.0, help='Hit rate threshold(s) for detecting important FP bits.')
    parser.add_argument('--out_prefix', type=str, default=None, help='Output file prefix. Default: None.')

    args = parser.parse_args()


    wd = Path(args.outdir_cff_top_batch)
    out_cff_top_batch_pth = wd / f"{args.out_prefix}_cfFP-top.msgpack"

    # init
    cffba = ConceptFromFingerprintBatchAnalysis(
        out_cff_top_batch_pth=out_cff_top_batch_pth,
        out_prefix=args.out_prefix,
        outdir_pth=wd,
    )
    
    cffba.analyze(threshold=args.threshold)


if __name__ == '__main__':
    main()