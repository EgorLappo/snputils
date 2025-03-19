import subprocess
from typing import List, Optional, Dict, Any
import os
import sys
import argparse
import shutil


class PongVizError(Exception):
    """Custom exception for Pong visualization errors"""
    pass


class FileMapError(Exception):
    """Custom exception for filemap creation errors"""
    pass


def pong_viz(folder_runs: str, output_dir: str, k: Optional[int] = None, min_k: Optional[int] = None, max_k: Optional[int] = None,
             runs: List[int] = None, run_prefix: str = 'train', ind2pop_path: Optional[str] = None,
             pop_names_path: Optional[str] = None, color_list_path: Optional[str] = None, verbose: bool = False) -> None:
    """
    Executes Pong visualization with the specified parameters.
    """

    # Check if pong is installed
    if not shutil.which('pong'):  # TODO: replace pong dependency with a custom implementation
        raise PongVizError("Pong is not installed. Please install it using 'pip install pong'")

    # Validate input parameters
    if not os.path.isdir(folder_runs):
        raise PongVizError(f"Input folder does not exist: {folder_runs}")
    if not output_dir:
        raise PongVizError("Output directory must be specified")
    os.makedirs(output_dir, exist_ok=True)

    # Optional files validation
    optional_files = {'ind2pop_path': ind2pop_path, 'pop_names_path': pop_names_path, 'color_list_path': color_list_path}
    for name, path in optional_files.items():
        if path and not os.path.isfile(path):
            raise PongVizError(f"Specified {name} file does not exist: {path}")

    try:
        filemap_path = create_filemap(folder_runs, k, min_k, max_k, runs, run_prefix)

        if verbose:
            print(f"Created filemap at: {filemap_path}")

        # Build command
        cmd = ["pong", "-m", filemap_path, "-o", output_dir]
        if ind2pop_path:
            cmd.extend(["-i", ind2pop_path])
        if pop_names_path:
            cmd.extend(["-n", pop_names_path])
        if color_list_path:
            cmd.extend(["-l", color_list_path])

        if verbose:
            print("Executing command:", " ".join(cmd))

        # Execute pong
        subprocess.run(cmd)

        if verbose:
            print("Pong execution successful")
            print("Results saved in:", output_dir)

    except FileMapError as e:
        raise PongVizError(f"Error creating filemap: {str(e)}")
    except subprocess.CalledProcessError as e:
        raise PongVizError(f"Error executing Pong: {str(e)}\nOutput: {e.output}")
    except Exception as e:
        raise PongVizError(f"Unexpected error: {str(e)}")


def create_filemap(folder: str, k: Optional[int] = None, min_k: Optional[int] = None, max_k: Optional[int] = None,
                   runs: List[int] = None, run_prefix: str = 'train_demo',) -> str:
    """
    Creates a filemap for training files organized by k values and runs and saves it to a file.

    Args:
        folder (str): Base folder path
        k (Optional[int]): Single k value to process. If specified, min_k and max_k are ignored
        min_k (Optional[int]): Minimum k value for range processing
        max_k (Optional[int]): Maximum k value for range processing
        runs (List[int]): List of run numbers
        run_prefix (str): Prefix for the run files (default: 'train_demo')

    Returns:
        str: Path to saved file

    Raises:
        FileMapError: If invalid parameters are provided or if configuration is incorrect
    """
    # Input validation
    if not folder:
        raise FileMapError("Folder path must be specified")
    if not runs:
        raise FileMapError("Runs list must be provided and non-empty")
    if not all(isinstance(run, int) and run > 0 for run in runs):
        raise FileMapError("All runs must be positive integers")
    if k is not None:
        if k < 1:
            raise FileMapError("k must be a positive integer")
        k_values = [k]
    else:
        if min_k is None or max_k is None:
            raise FileMapError("Either k or both min_k and max_k must be specified")
        if min_k > max_k:
            raise FileMapError("min_k cannot be greater than max_k")
        if min_k < 1:
            raise FileMapError("min_k must be a positive integer")
        k_values = range(min_k, max_k + 1)

    # Write file:
    filemap = {}
    try:
        for i, k_val in enumerate(k_values):
            for run_num in range(runs[i]):
                key = f'k{k_val}r{run_num+1}'
                if folder != '.':
                    path = folder + '/' + f'run{run_num+1}' + '/' + f'{run_prefix}.{k_val}.Q'
                else:
                    path = f'run{run_num+1}' + '/' + f'{run_prefix}.{k_val}.Q'

                filemap[key] = (k_val, path)

        filemap_path = os.path.join('pong_filemap')

        with open(filemap_path, 'w') as f:
            for i, (key, (k_val, path)) in enumerate(filemap.items()):
                f.write(f"{key}\t{k_val}\t{path}\n")

                if k_val in k_values:
                    index = k_values.index(k_val)
                    if (i + 1) % runs[index] == 0:
                        f.write("\n")

        return filemap_path

    except Exception as e:
        raise FileMapError(f"Error creating filemap: {str(e)}")


def parse_pong_args() -> Dict[str, Any]:
    """
    Parse command line arguments for pong visualization.

    Returns:
        Dict[str, Any]: Dictionary containing all parsed arguments ready for pong_viz

    Raises:
        SystemExit: If argument parsing fails or --help is used
    """
    parser = argparse.ArgumentParser(
        description='Execute Pong visualization with specified parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('folder_runs', help='Base folder containing the run files')
    parser.add_argument('output_dir', help='Directory where results will be saved')

    k_group = parser.add_mutually_exclusive_group(required=True)
    k_group.add_argument('-k', '--k', type=int, help='Single k value to process')
    k_group.add_argument('--k-range', nargs=2, type=int, metavar=('MIN_K', 'MAX_K'), help='Range of k values (min_k max_k)')

    parser.add_argument('--runs', type=int, nargs='+', help='Number of runs for each K')
    parser.add_argument('--prefix', default='train_demo', help='Prefix for run files')
    parser.add_argument('-i', '--ind2pop', help='Path to individual-to-population mapping file')
    parser.add_argument('-n', '--pop-names', help='Path to population names file')
    parser.add_argument('-c', '--color-list', help='Path to color list file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print detailed information during execution')
    args = parser.parse_args()

    params = {'folder_runs': args.folder_runs,
              'output_dir': args.output_dir,
              'runs': args.runs,
              'run_prefix': args.prefix,
              'verbose': args.verbose}

    if args.k is not None:
        params['k'] = args.k
    else:
        params['min_k'] = args.k_range[0]
        params['max_k'] = args.k_range[1]

    if args.ind2pop:
        params['ind2pop_path'] = args.ind2pop
    if args.pop_names:
        params['pop_names_path'] = args.pop_names
    if args.color_list:
        params['color_list_path'] = args.color_list

    return params


if __name__ == "__main__":
    try:
        params = parse_pong_args()
        pong_viz(**params)

    except (FileMapError, PongVizError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
