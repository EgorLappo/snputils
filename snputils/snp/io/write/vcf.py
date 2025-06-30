import logging
import pandas as pd
import numpy as np
import joblib
from typing import Union
from pathlib import Path

from snputils.snp.genobj import SNPObject

log = logging.getLogger(__name__)


class VCFWriter:
    """Writes a phenotype object in VCF format in the specified output path."""

    def __init__(self, snpobj: SNPObject, filename: str, n_jobs: int = -1, phased: bool = False):
        """
        Parameters
        ----------
        snpobj : ``SNPObject`` instance
            A SNPObject instance.
        file : str
            Path to write VCF file.
        n_jobs : int, default=-1
            Number of jobs to run in parallel. None means 1 unless in a
            joblib.parallel_backend context. -1 means using all processors.
        phased : bool, default=True
            If True, the genotype data is written in the "maternal|paternal"
            format. If False, the genotype data is written in the
            "maternal/paternal" format.
        """
        self.__snpobj = snpobj
        self.__filename = Path(filename)
        self.__n_jobs = n_jobs
        self.__phased = phased

    def write(
            self, 
            chrom_partition: bool = False,
            rename_missing_values: bool = True, 
            before: Union[int, float, str] = -1, 
            after: Union[int, float, str] = '.'
        ):
        """
        Writes the SNP data to VCF file(s).

        Args:
            chrom_partition (bool, optional):
                If True, individual VCF files are generated for each chromosome.
                If False, a single VCF file containing data for all chromosomes is created. Defaults to False.
            rename_missing_values (bool, optional):
                If True, renames potential missing values in `snpobj.calldata_gt` before writing. 
                Defaults to True.
            before (int, float, or str, default=-1): 
                The current representation of missing values in `calldata_gt`. Common values might be -1, '.', or NaN.
                Default is -1.
            after (int, float, or str, default='.'): 
                The value that will replace `before`. Default is '.'.
        """
        self.__chrom_partition = chrom_partition

        file_extensions = (".vcf", ".bcf")
        if self.__filename.suffix in file_extensions:
            self.__file_extension = self.__filename.suffix
            self.__filename = self.__filename.with_suffix('')
        else:
            self.__file_extension = ".vcf"

        # Optionally rename potential missing values in `snpobj.calldata_gt` before writing
        if rename_missing_values:
            self.__snpobj.rename_missings(before=before, after=after, inplace=True)

        data = self.__snpobj

        if self.__chrom_partition:
            chroms = data.unique_chrom

            for chrom in chroms:
                # Filter to include the data for the chromosome in particular
                data_chrom = data.filter_variants(chrom=chrom, inplace=False)

                log.debug(f'Storing chromosome {chrom}')
                self.write_chromosome_data(chrom, data_chrom)
        else:
            self.write_chromosome_data("All", data)

    def write_chromosome_data(self, chrom, data_chrom):
        """
        Writes the SNP data for a specific chromosome to a VCF file.

        Args:
            chrom: The chromosome name.
            data_chrom: The SNPObject instance containing the data for the chromosome.
        """
        # Obtain npy matrix with SNPs
        npy3 = data_chrom.calldata_gt  # shape: (n_windows, n_samples, 2)
        n_windows, n_samples, _ = npy3.shape

        # Keep sample names if appropriate
        data_samples = data_chrom.samples if len(data_chrom.samples) == n_samples else [get_name() for _ in range(n_samples)]


        # Format output file
        if chrom == "All":
            file = self.__filename.with_suffix(self.__file_extension)
        else:
            file = self.__filename.parent / f"{self.__filename.stem}_{chrom}{self.__file_extension}"

        # Write VCF file
        out = open(self.__filename.with_suffix(self.__file_extension), "w")
        # --- write VCF header ---
        out.write("##fileformat=VCFv4.1\n")
        out.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Phased Genotype">\n')
        
        for c in set(data_chrom.variants_chrom):
            out.write(f"##contig=<ID={c}>\n")
        cols = ["#CHROM","POS","ID","REF","ALT","QUAL","FILTER","INFO","FORMAT"] + list(data_chrom.samples)
        out.write("\t".join(cols) + "\n")
        
        sep = "|" if self.__phased else "/"
        for i in range(n_windows):
            chrom = data_chrom.variants_chrom[i]
            pos   = data_chrom.variants_pos[i]
            vid   = data_chrom.variants_id[i]
            ref   = data_chrom.variants_ref[i]
            alt   = data_chrom.variants_alt[i]
        
            # build genotype list per sample, one small array at a time
            row = npy3[i]  # shape: (n_samples, 2)
            genotypes = [
                f"{row[s,0]}{sep}{row[s,1]}"
                for s in range(n_samples)
            ]
        
            line = "\t".join([
                str(chrom), str(pos), vid, ref, alt,
                ".", "PASS", ".", "GT", *genotypes
            ])
            out.write(line + "\n")
        
        out.close()

def process_genotype(npy, i, n_snps, phased):
    """
    Process the genotype data for a particular individual in "maternal|paternal" 
    format for each SNP.

    Args:
        npy: Array containing genotype data for multiple individuals.
        i: Index of the individual to process.
        n_snps: Number of SNPs.

    Returns:
        genotype: List with "maternal|paternal" for each SNP.
    """
    # Get the genotype data for the specified individual's maternal and paternal SNPs
    maternal = npy[i*2, :].astype(str)     # maternal strand is the even row
    paternal = npy[i*2 + 1, :].astype(str)  # paternal strand is the odd row

    # Create a list with "maternal|paternal" format for each SNP
    sep = "|" if phased else "/"
    lst = [maternal, [sep] * n_snps, paternal]
    genotype = list(map(''.join, zip(*lst)))

    return genotype
