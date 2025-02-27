# Library Importation
import allel
import numpy as np
import pandas as pd
import sys
import time
import warnings
import logging
import datetime
from os import path
warnings.simplefilter("ignore", category=RuntimeWarning)


def process_vit(vit_file):
    """                                                                                       
    Process the Viterbi file to extract ancestry information.

    This function reads a Viterbi file containing ancestry information for individuals 
    at different genomic positions. It processes the file to construct an ancestry 
    matrix, where each row represents a genomic position and each column corresponds 
    to an individual.

    Args:
        vit_file (str): 
            Path to the Viterbi file. The file should be tab-separated, where the first 
            column represents genomic positions, and the remaining columns contain 
            ancestry assignments for individuals.

    Returns:
        np.ndarray of shape (n_snps, n_samples): 
            An ancestry matrix where `n_snps` represents the number of genomic positions 
            and `n_samples` represents the number of individuals.
    """
    start_time = time.time()
    vit_matrix = []
    with open(vit_file) as file:
        for x in file:
            x_split = x.replace('\n', '').split('\t')
            vit_matrix.append(np.array(x_split[1:-1]))
    ancestry_matrix = np.stack(vit_matrix, axis=0).T
    logging.info("VIT Processing Time: --- %s seconds ---" % (time.time() - start_time))
    
    return ancestry_matrix


def process_fbk(fbk_file, num_ancestries, prob_thresh):    
    """                                                                                       
    Process the FBK file to extract ancestry information.

    This function reads an FBK file containing ancestry probability values for 
    individuals across genomic positions. It processes these probabilities and 
    assigns ancestries based on a specified probability threshold.

    Args:
        fbk_file (str): 
            Path to the FBK file. The file should be space-separated, where each 
            row represents a genomic position, and columns contain probability values.
        num_ancestries (int): 
            Number of distinct ancestries in the dataset.
        prob_thresh (float): 
            Probability threshold for assigning ancestry to an individual at a 
            specific position.

    Returns:
        np.ndarray of shape (n_snps, n_samples): 
            An ancestry matrix where `n_snps` represents the number of genomic 
            positions and `n_samples` represents the number of individuals.
    """
    start_time = time.time()
    # Load FBK file into a DataFrame
    df_fbk = pd.read_csv(fbk_file, sep=" ", header=None)
    
    # Extract ancestry probability values, excluding the last column
    fbk_matrix = df_fbk.values[:, :-1]
    
    # Initialize ancestry matrix with zeros
    ancestry_matrix = np.zeros((fbk_matrix.shape[0], int(fbk_matrix.shape[1] / num_ancestries)), dtype=np.int8)
    
    # Assign ancestry based on probability threshold
    for i in range(num_ancestries):
        ancestry = i+1
        ancestry_matrix += (fbk_matrix[:, i::num_ancestries] > prob_thresh) * 1 * ancestry
    
    # Convert ancestry values to string format
    ancestry_matrix = ancestry_matrix.astype(str)
    logging.info("FBK Processing Time: --- %s seconds ---" % (time.time() - start_time))
    
    return ancestry_matrix


def process_tsv_fb(tsv_file, num_ancestries, prob_thresh, variants_pos, calldata_gt, variants_id):
    """                                                                                       
    Process the TSV/FB file to extract ancestry information.

    This function reads a TSV file containing ancestry probabilities, aligns 
    positions with the given genome data, and assigns ancestry labels based on 
    a probability threshold.

    Args:
        tsv_file (str): 
            Path to the TSV file. The file should be tab-separated and must contain 
            a 'physical_position' column along with ancestry probability values.
        num_ancestries (int): 
            Number of distinct ancestries in the dataset.
        prob_thresh (float): 
            Probability threshold for assigning ancestry to an individual at a 
            specific position.
        variants_pos (list of int): 
            A list containing the chromosomal positions for each SNP.
        calldata_gt (np.ndarray of shape (n_snps, n_samples)): 
            An array containing genotype data for each sample.
        variants_id (list of str): 
            A list containing unique identifiers (IDs) for each SNP.

    Returns:
        Tuple:
            - np.ndarray of shape (n_snps, n_samples): 
                An ancestry matrix indicating the ancestry assignment for each 
                individual at each genomic position.
            - np.ndarray of shape (n_snps, n_samples): 
                The updated genotype matrix after aligning with TSV positions.
            - list of str: 
                The updated list of SNP identifiers. 
    """
    start_time = time.time()
    # Load TSV file, skipping the first row
    df_tsv = pd.read_csv(tsv_file, sep="\t", skiprows=1)

    # Extract physical positions and remove unnecessary columns
    tsv_positions = df_tsv['physical_position'].tolist()
    df_tsv.drop(columns = ['physical_position', 'chromosome', 'genetic_position', 'genetic_marker_index'], inplace=True)

    # Convert DataFrame to NumPy array
    tsv_matrix = df_tsv.values

    # Find the range of positions that match between TSV and provided positions
    i_start = variants_pos.index(tsv_positions[0])
    if tsv_positions[-1] in variants_pos:
        i_end = variants_pos.index(tsv_positions[-1]) + 1
    else:
        i_end = len(variants_pos)

    # Update genome data to match the TSV file range
    calldata_gt = calldata_gt[i_start:i_end, :]
    variants_pos = variants_pos[i_start:i_end]
    variants_id = variants_id[i_start:i_end]

    # Initialize probability matrix with the same shape as filtered positions
    prob_matrix = np.zeros((len(variants_pos), tsv_matrix.shape[1]), dtype=np.float16)

    # Align TSV probabilities with genomic positions
    i_tsv = -1
    next_pos_tsv = tsv_positions[i_tsv+1]
    for i in range(len(variants_pos)):
        pos = variants_pos[i]
        if pos >= next_pos_tsv and i_tsv + 1 < tsv_matrix.shape[0]:
            i_tsv += 1
            probs = tsv_matrix[i_tsv, :]
            if i_tsv + 1 < tsv_matrix.shape[0]:
                next_pos_tsv = tsv_positions[i_tsv+1]
        prob_matrix[i, :] = probs

    # Replace TSV matrix with aligned probability matrix
    tsv_matrix = prob_matrix

    # Initialize ancestry matrix
    ancestry_matrix = np.zeros((tsv_matrix.shape[0], int(tsv_matrix.shape[1] / num_ancestries)), dtype=np.int8)

    # Assign ancestry based on probability threshold
    for i in range(num_ancestries):
        ancestry = i+1
        ancestry_matrix += (tsv_matrix[:, i::num_ancestries] > prob_thresh) * 1 * ancestry

    # Adjust ancestry values to start at 0
    ancestry_matrix -= 1

    # Convert ancestry matrix to string format
    ancestry_matrix = ancestry_matrix.astype(str)
    logging.info("TSV Processing Time: --- %s seconds ---" % (time.time() - start_time))
    
    return ancestry_matrix, calldata_gt, variants_id


def process_laiobj(laiobj, variants_pos, variants_chrom, calldata_gt, variants_id):
    """                                                                                       
    Process the LocalAncestryObject to extract ancestry information.

    This function processes a LocalAncestryObject containing ancestry segment data. 
    It aligns the ancestry data with the provided genomic positions and 
    assigns ancestry labels accordingly.

    Args:
        laiobj (dict): 
            A LocalAncestryObject instance.
        variants_pos (list of int): 
            A list containing the chromosomal positions for each SNP.
        variants_chrom (list of int): 
            A list containing the chromosome for each SNP.
        calldata_gt (np.ndarray of shape (n_snps, n_samples)): 
            An array containing genotype data for each sample.
        variants_id (list of str): 
            A list containing unique identifiers (IDs) for each SNP.

    Returns:
        Tuple:
            - np.ndarray of shape (n_snps, n_samples): 
                An ancestry matrix where `n_snps` represents the number of genomic 
                positions and `n_samples` represents the number of individuals. 
                Ancestry values are assigned based on LAI data.
            - np.ndarray of shape (n_snps, n_samples): 
                The updated genotype matrix after aligning with LAI positions.
            - list of str: 
                The updated list of SNP identifiers.                                                                                             
    """
    start_time = time.time()
    
    # Extract start and end positions of ancestry segments
    tsv_spos = laiobj['physical_pos'][:,0].tolist()
    tsv_epos = laiobj['physical_pos'][:,1].tolist()
    
    # Extract ancestry matrix from LAI object
    tsv_matrix = laiobj['lai']

    # Determine index range for matching positions
    i_start = variants_pos.index(tsv_spos[0])
    if tsv_epos[-1] in variants_pos:
        i_end = variants_pos.index(tsv_epos[-1])
    else:
        i_end = len(variants_pos)
    
    # Update genotype matrix and associated metadata
    calldata_gt = calldata_gt[i_start:i_end, :]
    variants_pos = variants_pos[i_start:i_end]
    variants_id = variants_id[i_start:i_end]

    # Extract chromosome information from LAI object
    tsv_chromosomes = laiobj['chromosomes']

     # Initialize ancestry matrix
    ancestry_matrix = np.zeros((len(variants_pos), tsv_matrix.shape[1]), dtype=np.int8)

    # Variables for iterating through ancestry data
    i_tsv = -1
    next_pos_tsv = tsv_spos[i_tsv+1]
    next_chrom_tsv = tsv_chromosomes[i_tsv+1]

    # Assign ancestry based on genomic position and chromosome alignment
    for i, pos in enumerate(variants_pos):
        if pos >= next_pos_tsv and int(variants_chrom[i]) == int(next_chrom_tsv) and i_tsv + 1 < tsv_matrix.shape[0]:
            i_tsv += 1
            ancs = tsv_matrix[i_tsv, :]
            if i_tsv + 1 < tsv_matrix.shape[0]:
                next_pos_tsv = tsv_spos[i_tsv+1]
                next_chrom_tsv = tsv_chromosomes[i_tsv+1]

        ancestry_matrix[i, :] = ancs

    # Convert ancestry matrix to string format
    ancestry_matrix = ancestry_matrix.astype(str)
    
    logging.info("TSV Processing Time: --- %s seconds ---" % (time.time() - start_time))
    return ancestry_matrix, calldata_gt, variants_id


def process_beagle(beagle_file, rs_ID_dict, rsid_or_chrompos):
    """                                                                                       
    Process a Beagle file to extract genotype and variant information.

    This function processes a Beagle genotype file to extract individual IDs, 
    reformat variant identifiers, and encode genetic information into a structured 
    genotype matrix.

    Args:
        beagle_file (str): 
            Path to the Beagle file containing genotype data.
        rs_ID_dict (dict): 
            Dictionary mapping variant identifiers to their reference alleles. 
            If an identifier is not found, it will be added to the dictionary.
        rsid_or_chrompos (int): 
            Format specification for variant identifiers:
            - `1`: Use rsID format.
            - `2`: Use Chromosome_Position format.

    Returns:
        Tuple:
            - np.ndarray of shape (n_snps, n_samples * ploidy): 
                A genotype matrix where `n_snps` represents the number of genomic 
                positions and `n_samples * ploidy` represents the flattened genotype calls 
                for diploid individuals.
            - np.ndarray of shape (n_samples,): 
                Array of individual IDs corresponding to the genotype matrix.
            - list of int or float: 
                List of variant identifiers, formatted based on `rsid_or_chrompos` selection.
            - dict: 
                Updated reference allele dictionary mapping variant identifiers to reference alleles.
    """
    start_time = time.time()
    variants_id = []
    lis_beagle = []
    with open(beagle_file) as file:
        # Read header line and extract individual IDs
        x = file.readline()
        x_split = x.replace('\n', '').split('\t')
        ind_IDs = x_split[2:]
        ind_IDs = np.array(ind_IDs)
        for x in file:
            x_split = x.replace('\n', '').split('\t')
            if rsid_or_chrompos == 1:
                variants_id.append(int(x_split[1][2:]))
            elif rsid_or_chrompos == 2:
                rs_ID_split = x_split[1].split('_')
                variants_id.append(np.float64(rs_ID_split[0] + '.' + rs_ID_split[1][::-1]))
            else:
                sys.exit("Illegal value for rsid_or_chrompos. Choose 1 for rsID format or 2 for Chromosome_position format.")
            lis_beagle.append(x_split[2:])
    
    # Initialize genotype matrix
    calldata_gt = np.zeros((len(lis_beagle),len(lis_beagle[0])), dtype=np.float16)
    
    # Process reference allele encoding
    processed_IDs = rs_ID_dict.keys()
    for i in range(len(lis_beagle)):
        # Check how we usually encode:
        if (variants_id[i] in processed_IDs):
            ref = rs_ID_dict[variants_id[i]]
        else:
            ref = lis_beagle[i][0]
            rs_ID_dict[variants_id[i]] = ref

        for j in range(1, len(lis_beagle[i])):
            calldata_gt[i, j] = (lis_beagle[i][j] != ref)*1

    logging.info("Beagle Processing Time: --- %s seconds ---" % (time.time() - start_time))

    return calldata_gt, ind_IDs, variants_id, rs_ID_dict


def process_vcf(snpobj, rs_ID_dict, rsid_or_chrompos):
    """                                                                                       
    Process a VCF file to extract genotype and variant information.

    This function processes a VCF file to extract genotypic data, reformat variant identifiers, 
    and encode genetic information into a structured genotype matrix.

    Args:
        snpobj (dict): 
            A SNPObject instance.
        rs_ID_dict (dict): 
            Dictionary mapping variant identifiers to their reference alleles. 
            If an identifier is not found, it will be added to the dictionary.
        rsid_or_chrompos (int): 
            Format specification for variant identifiers:
            - `1`: Use rsID format.
            - `2`: Use Chromosome_Position format.

    Returns:
        Tuple:
            - np.ndarray of shape (n_snps, n_samples * ploidy): 
                A genotype matrix where `n_snps` represents the number of genomic 
                positions and `n_samples * ploidy` represents the flattened genotype calls 
                for diploid individuals.
            - np.ndarray of shape (n_samples * ploidy,): 
                Array of individual IDs corresponding to the genotype matrix.
            - list of int or float: 
                List of variant identifiers, formatted based on `rsid_or_chrompos` selection.
            - list of int: 
                List of genomic positions corresponding to the variants.
            - dict: 
                Updated reference allele dictionary mapping variant identifiers to reference alleles.
    """
    start_time = time.time()

    # Extract genotype data and reshape to 2D (n_snps, n_samples * ploidy)
    calldata_gt = snpobj['calldata_gt']
    n_snps, n_samples, ploidy = calldata_gt.shape
    calldata_gt = calldata_gt.reshape(n_snps, n_samples * ploidy).astype(np.float16)

    # Replace missing genotype values (-1) with NaN
    np.place(calldata_gt, calldata_gt < 0, np.nan)

    # Extract variant identifiers based on rsID format selection
    if rsid_or_chrompos == 1:
        IDs = snpobj['variants_id']
        variants_id = [int(x[2:]) for x in IDs]
    elif rsid_or_chrompos == 2:
        variants_id = []
        for i in range(len(snpobj['variants_chrom'])):
            variants_id.append(np.float64(snpobj['variants_chrom'][i] + '.' + str(snpobj['variants_pos'][i])[::-1]))
    else:
        sys.exit("Illegal value for rsid_or_chrompos. Choose 1 for rsID format or 2 for Chromosome_position format.")

    # Extract reference alleles and individual sample IDs
    ref_vcf = snpobj['variants_ref']
    samples = snpobj['samples']

    # Generate individual IDs for diploid samples
    ind_IDs = np.array([f"{sample}_{suffix}" for sample in samples for suffix in ["A", "B"]])
    
    # Extract variant positions
    positions = snpobj['variants_pos'].tolist()
    
    # Process reference allele encoding
    for i, (rs_ID, ref_val) in enumerate(zip(variants_id, ref_vcf)):
        # If rs_ID is not in the dictionary, it is set to ref_val
        # Otherwise, it returns the existing value
        ref = rs_ID_dict.setdefault(rs_ID, ref_val)

        # Flip genotype encoding if reference allele differs from stored reference
        if ref != ref_val:
            calldata_gt[i, :] = 1 - calldata_gt[i, :]
    
    logging.info("VCF Processing Time: --- %s seconds ---" % (time.time() - start_time))
    return calldata_gt, ind_IDs, variants_id, positions, rs_ID_dict


def average_parent_snps(masked_ancestry_matrix):
    """                                                                                       
    Average haplotypes to obtain genotype data for individuals. 
    
    This function combines pairs of haplotypes by computing their mean.

    Args:
        masked_ancestry_matrix (np.ndarray of shape (n_snps, n_haplotypes)): 
            The masked matrix for an ancestry, where `n_snps` represents the number of SNPs and 
            `n_haplotypes` represents the number of haplotypes.

    Returns:
        np.ndarray of shape (n_snps, n_samples):  
            A new matrix where each pair of haplotypes has been averaged, resulting in genotype 
            data for individuals instead of haplotypes.
    """
    start = time.time()
    # Initialize a new matrix with half the number of columns
    new_matrix = np.zeros((masked_ancestry_matrix.shape[0], int(masked_ancestry_matrix.shape[1]/2)), dtype = np.float16)
    
    # Iterate through haplotype pairs, computing the mean for each pair
    for i in range(0, masked_ancestry_matrix.shape[1], 2):
        new_matrix[:, int(i/2)] = np.nanmean(masked_ancestry_matrix[:,i:i+2],axis=1, dtype = np.float16)
    logging.info("Combining time --- %s seconds ---" % (time.time() - start))
    
    return new_matrix


def mask(ancestry_matrix, calldata_gt, unique_ancestries, ancestry_int_list, average_strands = False):
    """                                                                                       
    Mask genotype data based on ancestry labels.

    This function applies ancestry-based masking to a genomic matrix, selectively retaining genotype 
    data for individual haplotypes based on their assigned ancestry.

    Args:
        ancestry_matrix (np.ndarray of shape (n_snps, n_haplotypes)): 
            Matrix indicating the ancestry assignments, where `n_snps` represents the number of SNPs and 
            `n_haplotypes` represents the number of haplotypes.
        calldata_gt (np.ndarray of shape (n_snps, n_haplotypes)): 
            Genetic matrix encoding the genotype information for haplotypes.
        unique_ancestries (list): 
            List of distinct ancestries present in the dataset (e.g., ['0', '1', '2']).
        ancestry_int_list (list): 
            List of ancestry identifiers (e.g., ['0', '1', '2']).
        average_strands (bool, optional): 
            Whether to average haplotypes for each individual. Defaults to `False`.

    Returns:
        dict: 
            Dictionary containing masked genetic matrices for each ancestry group.
    """
    start_time = time.time()

    # Dictionary to store masked matrices for each ancestry group
    masked_matrices = {}

    # Iterate through each unique ancestry
    for i in range(len(unique_ancestries)):
        # Get the ancestry identifier
        ancestry = unique_ancestries[i] 
        # Get the corresponding dictionary key
        dict_ancestry = ancestry_int_list[i]

        # Initialize an empty masked array with NaN values
        masked = np.empty(ancestry_matrix.shape[0] * ancestry_matrix.shape[1], dtype = np.float16)
        masked[:] = np.nan

        # Identify positions in the ancestry matrix that match the current ancestry
        arg = ancestry_matrix.reshape(-1) == ancestry

        # Apply masking: retain genotype data only for the matched ancestry
        masked[arg] = calldata_gt.reshape(-1)[arg]
        logging.info("Masking for ancestry " + str(ancestry) + " --- %s seconds ---" % (time.time() - start_time))

        # If averaging strands is enabled, average the SNPs for each individual
        if (average_strands == True):
            masked_matrices[dict_ancestry] = average_parent_snps(masked.reshape(ancestry_matrix.shape).astype(np.float16))    
        else:
            # Otherwise, store the masked matrix as is
            masked_matrices[dict_ancestry] = masked.reshape(ancestry_matrix.shape).astype(np.float16)

        start_time = time.time()

    return masked_matrices


def get_masked_matrix(snpobj, beagle_or_vcf, laiobj, vit_or_fbk_or_fbtsv_or_msptsv,
                      is_mixed, is_masked, num_ancestries, average_strands, rs_ID_dict, rsid_or_chrompos):
    """                                                                                       
    Input Parameter Parser                                                
                                                                                               
    Parameters                                                                                
    ----------       
    snpobj : string with the path of our beagle/vcf file.  
    beagle_or_vcf       : int
                          indicates file type 1=beagle, 2=vcf
                          
    vit_fbk_tsv_filename: string with the path of our vit/fbk/tsv file.
    vit_or_fbk_or_tsv   : int
                          indicates file type 1=vit, 2=fbk, 3=tsv
    fb_or_msp           : int
                          indicates 1=fb, 2=msp
    is_masked           : boolean
                          indicates if output matrix needs to be masked.
    num_ancestries      : int
                          number of distinct ancestries in dataset
    average_strands     : boolean
                          indicates whether to combine haplotypes for each individuals.
    prob_thresh         : float  
                          probability threshold for ancestry assignment.
    rs_ID_dict          : dictionary showing the previous encoding for a specific 
                          rs ID.

    Returns                                                                                   
    -------   
    masked_matrices  : (m, n)/(m, n, num_ancestries) array/dictionary
                       unmasked matrix/masked matrices for each of the distinct ancestries in the 
                       dataset.
    ind_IDs          : (n,) array
                       Individual IDs for all individuals in the matrix. 
    variants_id           : (m,) array
                       rs IDs of all the positions included in our matrix. 
    rs_ID_dict       :
                       Encoding dictionary for each of the positions in dataset.             
    """
    if beagle_or_vcf == 1:
        calldata_gt, ind_IDs, variants_id, rs_ID_dict = process_beagle(snpobj, rs_ID_dict, rsid_or_chrompos)
    elif beagle_or_vcf == 2:
        calldata_gt, ind_IDs, variants_id, positions, rs_ID_dict = process_vcf(snpobj, rs_ID_dict, rsid_or_chrompos)
        
    if is_masked and vit_or_fbk_or_fbtsv_or_msptsv != 0:
        
        ancestry_matrix, calldata_gt, variants_id = process_laiobj(laiobj, positions, snpobj['variants_chrom'], calldata_gt, variants_id)

        if vit_or_fbk_or_fbtsv_or_msptsv == 1 or vit_or_fbk_or_fbtsv_or_msptsv == 2:
            unique_ancestries = [str(i) for i in np.arange(1, num_ancestries+1)]
        else:
            unique_ancestries = [str(i) for i in np.arange(0, num_ancestries)]
        if is_mixed:
            ancestry_int_list = [str(i) for i in np.arange(0, num_ancestries)]
        else:
            ancestry_int_list = unique_ancestries
        masked_matrices = mask(ancestry_matrix, calldata_gt, unique_ancestries, ancestry_int_list, average_strands)
    
    else:
        if not is_masked:
            ancestry_int_list = [str(i) for i in np.arange(1, num_ancestries+1)]
        elif is_mixed or beagle_or_vcf == 2:
            ancestry_int_list = [str(i) for i in np.arange(0, num_ancestries)]
        else:
            ancestry_int_list = [str(i) for i in np.arange(1, num_ancestries+1)]
        masked_matrices = {}
        if average_strands:
            calldata_gt_avg = average_parent_snps(calldata_gt)
            for ancestry in ancestry_int_list:
                masked_matrices[ancestry] = calldata_gt_avg
        else:
            for ancestry in ancestry_int_list:
                masked_matrices[ancestry] = calldata_gt
        logging.info("No masking")
        
    return masked_matrices, ind_IDs, variants_id, rs_ID_dict


def array_process(snpobj, laiobj, average_strands, prob_thresh, is_masked, rsid_or_chrompos, num_arrays=1): 
    """                                                                                       
    Dataset processing of each of the individual arrays.                                               
                                                                                               
    Parameters                                                                                
    ----------                                                                                
    beagle_vcf_file  : string
                       Beagle/VCF Filename defined by user.
    vit_fbk_tsv_file : string
                       Viterbi/TSV/FBK Filename defined by user.
    beagle_or_vcf    : int
                       indicates file type 1=beagle, 2=vcf
    vit_or_fbk_or_tsv: int
                       indicates file type 1=vit, 2=fbk, 3=tsv
    fb_or_msp        : int
                       indicates 1=fb, 2=msp     
    num_arrays       : Total number of arrays in dataset.
    num_ancestries   : Number of unique ancestries in dataset. 
    average_strands  : boolean
                       Indicates whether to combine haplotypes for each individual.
    prob_thresh      : float  
                       Probability threshold for ancestry assignment.
    is_masked        : boolean
                       indicates if output matrix needs to be masked. 
    save_masks       : boolean
                       indicates if mask files needs to be saved.
    masks_file       : string
                       npz filename defined by user to save the mask files.
                                                                                   
    Returns                                                                                   
    -------                                                                                   
    masks      : (num_arrays, ) list                                                                          
                 List of masked matrices for each ancestries at each given array.
    rs_ID_list : (num_arrays, ) list 
                 List of rs IDs for each of the processed arrays.
    ind_ID_list: 
                 List of individual IDs for each of the processed arrays.

    """
    beagle_or_vcf_list = [2]
    vit_or_fbk_or_fbtsv_or_msptsv_list = [4]

    if (1 in beagle_or_vcf_list) and (2 in beagle_or_vcf_list):
        is_mixed = True
    else:
        is_mixed = False

    # Initialization:
    rs_ID_dict = {}
    masks =[]
    rs_ID_list = []
    ind_ID_list = []

    # Obtain number of ancestries in LAI object
    num_ancestries = laiobj.n_ancestries

    for i in range(num_arrays):
        logging.info("------ Array "+ str(i+1) + " Processing: ------")
        genome_matrix, ind_IDs, variants_id, rs_ID_dict = get_masked_matrix(snpobj, beagle_or_vcf_list[i],
                                                                       laiobj,
                                                                       vit_or_fbk_or_fbtsv_or_msptsv_list[i], is_mixed, is_masked,
                                                                       num_ancestries, average_strands, rs_ID_dict,
                                                                       rsid_or_chrompos)


        masks.append(genome_matrix)
        rs_ID_list.append(variants_id)
        if (average_strands == False):
            ind_ID_list.append(ind_IDs)
        else:
            ind_ID_list.append(remove_AB_indIDs(ind_IDs))
        
    return masks, rs_ID_list, ind_ID_list


def remove_AB_indIDs(ind_IDs):
    new_ind_IDs = []
    for i in range(int(len(ind_IDs)/2)):
        new_ind_IDs.append(ind_IDs[2*i][:-2])
    new_ind_IDs = np.array(new_ind_IDs)
    return new_ind_IDs


def add_AB_indIDs(ind_IDs):
    new_ind_IDs = []
    for i in range(len(ind_IDs)):
        new_ind_IDs.append(str(ind_IDs[i]) + '_A')
        new_ind_IDs.append(str(ind_IDs[i]) + '_B')
    new_ind_IDs = np.array(new_ind_IDs)
    return new_ind_IDs


def process_labels_weights(labels_file, masks, rs_ID_list, ind_ID_list, average_strands, ancestry, min_percent_snps, remove_labels_dict, is_weighted, save_masks, masks_file, num_arrays=1):
    labels_df = pd.read_csv(labels_file, sep='\t')
    labels_df['indID'] = labels_df['indID'].astype(str)
    label_list = []
    weight_list = []
    for array_ind in range(num_arrays):
        masked_matrix = masks[array_ind][ancestry]
        ind_IDs = ind_ID_list[array_ind]
        if average_strands:
            labels = np.array(labels_df['label'][labels_df['indID'].isin(ind_IDs)])
            label_ind_IDs = np.array(labels_df['indID'][labels_df['indID'].isin(ind_IDs)])
        else:
            temp_ind_IDs = remove_AB_indIDs(ind_IDs)
            labels = np.array(labels_df['label'][labels_df['indID'].isin(temp_ind_IDs)])
            labels = np.repeat(labels, 2)
            label_ind_IDs = np.array(labels_df['indID'][labels_df['indID'].isin(temp_ind_IDs)])
            label_ind_IDs = add_AB_indIDs(label_ind_IDs)
        keep_indices = [ind_IDs.tolist().index(x) for x in label_ind_IDs]
        masked_matrix = masked_matrix[:,keep_indices]
        ind_IDs = ind_IDs[keep_indices]
        array_num = array_ind + 1
        if not is_weighted:
            weights = np.ones(len(labels))
            combinations = np.zeros(len(labels))
            combination_weights = np.zeros(len(labels))
        else:
            if average_strands:
                weights = np.array(labels_df['weight'][labels_df['indID'].isin(ind_IDs)])
                if 'combination' in labels_df.columns:
                    combinations = np.array(labels_df['combination'][labels_df['indID'].isin(ind_IDs)])
                else:
                    combinations = np.zeros(len(weights))
                if 'combination_weight' in labels_df.columns:
                    combination_weights = np.array(labels_df['combination_weight'][labels_df['indID'].isin(ind_IDs)])
                else:
                    combination_weights = np.ones(len(weights))
            else:
                temp_ind_IDs = remove_AB_indIDs(ind_IDs)
                weights = np.array(labels_df['weight'][labels_df['indID'].isin(temp_ind_IDs)])
                weights = np.repeat(weights, 2)
                if 'combination' in labels_df.columns:
                    combinations = np.array(labels_df['combination'][labels_df['indID'].isin(temp_ind_IDs)])
                    combinations = np.repeat(combinations, 2)
                else:
                    combinations = np.zeros(len(weights))
                if 'combination_weight' in labels_df.columns:
                    combination_weights = np.array(labels_df['combination_weight'][labels_df['indID'].isin(temp_ind_IDs)])
                    combination_weights = np.repeat(combination_weights, 2)
                else:
                    combination_weights = np.ones(len(weights))
        if array_num in remove_labels_dict:
            remove_labels = remove_labels_dict[array_num]
            for i in range(len(labels)):
                if labels[i] in remove_labels:
                    weights[i] = 0
        percent_snps = 100 * (1 - np.mean(np.isnan(masked_matrix), axis=0))
        keep_indices = np.argwhere(percent_snps >= min_percent_snps).flatten()
        masked_matrix = masked_matrix[:,keep_indices]
        ind_IDs = ind_IDs[keep_indices]
        labels = labels[keep_indices]
        weights = weights[keep_indices]
        combinations = combinations[keep_indices]
        combination_weights = combination_weights[keep_indices]
        keep_indices = np.argwhere(weights > 0).flatten()
        masked_matrix_new = masked_matrix[:,keep_indices]
        ind_IDs_new = ind_IDs[keep_indices]
        labels_new = labels[keep_indices]
        weights_new = weights[keep_indices]
        pos_combinations = sorted(set(combinations[combinations > 0]))
        num_combinations = len(pos_combinations)
        if num_combinations > 0:
            for combination in pos_combinations:
                combined_indices = np.argwhere(combinations == combination)
                combined_col = np.nanmean(masked_matrix[:,combined_indices], axis=1)
                masked_matrix_new = np.append(masked_matrix_new, combined_col, axis=1)
                ind_IDs_new = np.append(ind_IDs_new, 'combined_ind_' + str(combination))
                labels_new = np.append(labels_new, labels[combined_indices[0][0]])
                weights_new = np.append(weights_new, combination_weights[combined_indices[0][0]])
        masked_matrix = masked_matrix_new
        ind_IDs = ind_IDs_new
        labels = labels_new
        weights = weights_new
        masks[array_ind][ancestry] = masked_matrix
        ind_ID_list[array_ind] = ind_IDs
        label_list += labels.tolist()
        weight_list += weights.tolist()
    label_list = np.array(label_list)
    weight_list = np.array(weight_list)
    if save_masks:
        np.savez_compressed(masks_file, masks=masks, rs_ID_list=rs_ID_list, ind_ID_list=ind_ID_list,
                 labels=label_list, weights=weight_list, protocol=4)
    return masks, ind_ID_list, label_list, weight_list


def center_masked_matrix(masked_matrix):
    masked_matrix -= np.nanmean(masked_matrix, axis=0)
    return masked_matrix


def logger_config(verbose=True):
    logging_config = {"version": 1, "disable_existing_loggers": False}
    fmt = '[%(levelname)s] %(asctime)s: %(message)s'
    logging_config["formatters"] = {"basic": {"format": fmt, "datefmt": "%Y-%m-%d %H:%M:%S"}}
    now = datetime.datetime.now()

    logging_config["handlers"] = {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG" if verbose else "INFO",
                "formatter": "basic",
                "stream": "ext://sys.stdout"
            },
            "info_file_handler": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG" if verbose else "INFO", 
                "formatter": "basic",
                "maxBytes": 10485760,
                "backupCount": 20,
                "filename": f"log{now.year}_{now.month}_{now.day}__{now.hour}_{now.minute}.txt", # choose a better name or name as param?
                "encoding": "utf8"
                }
            }
    logging_config["root"] = {
                "level": "DEBUG",
                "handlers": ["console", "info_file_handler"]
            }
    return logging_config
