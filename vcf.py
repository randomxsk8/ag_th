#! /opt/conda/bin/python

__author__ = 'Nicholas Harding'

# This script takes an hdf5 file, reads the object in and produces a minimal
# vcf file. Really minimal, just the essential 8 columns and genotypes.

import argparse
import h5py
import allel
import numpy as np
import gzip
import pandas as pd
import os
import gcsfs
import zarr
from itertools import compress

gcs_bucket_fs = gcsfs.GCSFileSystem(project='malariagen-jupyterhub', token='anon', access='read_only')

geno_bi_path = os.path.join("ag1000g-release/phase2.AR1/variation/main/zarr/biallelic/ag1000g.phase2.ar1.pass.biallelic")
gcsacmap = gcs_bucket_fs.get_mapper(root=geno_bi_path)
callset_biallel= zarr.Group(gcsacmap, read_only=True)
metadata = pd.read_csv("samples.meta.txt", sep="\t")
pop_selection = metadata.population.isin({'GHcol', 'GHgam', 'BFgam', 'BFcol', 'GM', 'GW', 'GNgam', 'GNcol',
       'CIcol'}).values
callset_fn = callset_biallel

def get_consecutive_true(a):
    if a.sum() == 0:
        return 0
    else:
        return np.diff(np.where(
            np.concatenate(([a[0]], a[:-1] != a[1:], [True])))[0])[::2].max()

chunk_size = 100000

parser = argparse.ArgumentParser(
    description='Tool to produce a vcf file from an hdf5 file')

parser.add_argument('input', help='input hdf5 file')
parser.add_argument('output', help='output file stem')

parser.add_argument('--filtermissing', '-F', action='store_false', default=True,
                    dest='keepmissing')
parser.add_argument('--cutoff', '-C', action='store', default=0.04,
                    dest='missingcutoff', type=float,
                    help='Maximum missing GTs tolerated in a sample')
parser.add_argument('--pedigree', '-P', action='store', dest='pedigree',
                    help='path to load pedigree file')

# to do: add option to only filter individual crosses.
args = parser.parse_args()

with h5py.File(args.input, mode='r') as h5_handle:
    with gzip.open(args.output + '.vcf.gz', 'wb') as f:

        f.write(b'##fileformat=VCFv4.1\n')
        f.write(b'##FORMAT=<ID=GT,Number=1,Type=String,'
                b'Description="Genotype">\n')
        f.write(b'##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count'
                b'in genotypes, for each ALT allele, in the same order as'
                b'listed">\n')

        f.write(b'##contig=<ID=2L,length=49364325>\n')
        f.write(b'##contig=<ID=2R,length=61545105>\n')
        f.write(b'##contig=<ID=3L,length=41963435>\n')
        f.write(b'##contig=<ID=3R,length=53200684>\n')
        f.write(b'##contig=<ID=UNKN,length=42389979>\n')
        f.write(b'##contig=<ID=X,length=24393108>\n')
        f.write(b'##contig=<ID=Y_unplaced,length=237045>\n')
        f.write(b'##reference=file:///data/anopheles/ag1000g/data/genome/AgamP3'
                b'/Anopheles-gambiae-PEST_CHROMOSOMES_AgamP3.fa\n')

        reqd = [b'#CHROM', b'POS', b'ID', b'REF', b'ALT',
                b'QUAL', b'FILTER', b'INFO', b'FORMAT']

        # rememeber to act on all 1st level keys!
        # does not support multiple chromosomes currently!
        # Actually should probably add to filter script...
        assert len(h5_handle.keys()) <= 1
        for k in h5_handle.keys():

            fh_samples = [str(s) for s in callset_fn['3R']["samples"][:]]
            samples = list(compress(fh_samples, pop_selection))            
            missing_rates = np.zeros(len(samples))
            ok_samples = np.ones(len(samples), dtype="bool")

            gt = allel.GenotypeChunkedArray(
                h5_handle[k][:])

            if not args.keepmissing:

                missing_gt = gt.is_missing()

                for i, s in enumerate(samples):

                    consecutive_miss = get_consecutive_true(missing_gt[:, i])
                    miss_rate_i = consecutive_miss/float(missing_gt.shape[0])

                    print("Missing rate of", s, ':',
                          "{:.8f}".format(miss_rate_i),
                          "({0}/{1})".format(i+1, len(samples)))
                    missing_rates[i] = miss_rate_i

                print("Rate max:", missing_rates.max())
                ok_samples = missing_rates < args.missingcutoff

                if np.any(~ok_samples):
                    msg = "The following {0} samples are excluded as they " \
                          "have a consecutive missing gt run of >= {1} of " \
                          "all calls:".format(str(np.sum(~ok_samples)),
                                              str(args.missingcutoff))
                    print(msg)

                    for sa, rt in zip(
                            np.compress(~ok_samples, samples).tolist(),
                            np.compress(~ok_samples, missing_rates).tolist()):
                        print(sa + ": " + str(rt))

                    samples = [s.decode() for s in np.compress(
                        ok_samples, samples).tolist()]
                else:
                    print("All samples meet the missingness run threshold ({0})"
                          .format(str(args.missingcutoff)))

            if args.pedigree is not None:
                phasing.utils.create_samples_file(args.pedigree,
                                                  args.output + '.sample',
                                                  samples)

            f.write(b"\t".join(reqd + samples) + b"\n")

            number_variants = h5_handle[k]['variants']['POS'][:].size
            chunks = np.arange(0, number_variants + chunk_size, chunk_size)
            assert chunks.max() > number_variants

            for start, stop in zip(chunks[:-1], chunks[1:]):
                sl = slice(start, stop)
                positions = h5_handle[k]['variants']['POS'][sl]
                reference = h5_handle[k]['variants']['REF'][sl]
                alternate = h5_handle[k]['variants']['ALT'][sl]
                genotypes = h5_handle[k]['calldata']['genotype'][sl]
                genotypes = np.compress(ok_samples, genotypes, axis=1)
                multiple_alts = alternate.ndim > 1

                for pos, ref, alt, gt in zip(positions, reference,
                                             alternate, genotypes):
                    filterstring = 'PASS'
                    # This line filters variants where ALL genotypes are missing
                    if not args.keepmissing and np.all(gt == -1):
                        continue

                    # alt may be an np array, with several entries.
                    if multiple_alts:
                        alt = b",".join(x for x in alt if x != b'')

                    try:
                        gstr = np.apply_along_axis(b"/".join, axis=1,
                                                   arr=gt.astype("a2"))

                        genotype_str = b"\t".join([s for s in gstr]) + b"\n"
                        genotype_str = genotype_str.replace(b"-1/-1", b"./.")

                        line = b"\t".join(
                            [k.encode(), str(pos).encode()] +
                            [b'.', ref, alt, b'0', b'.', b'.', b'GT'] +
                            [genotype_str])

                        f.write(line)

                    except TypeError:
                        print(pos)
                        print(ref)
                        print(alt)
                        print(gt)
                        raise TypeError("Some data wasn't of the correct type.")
