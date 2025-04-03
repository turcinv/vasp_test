import csv
import gc
import logging
import multiprocessing as mp
import os
import pickle
import sys
import threading
import time
from collections import defaultdict
from typing import List, Tuple, Any, Dict, Union
import mdtraj as md
import numpy as np
import psutil
from colorama import Fore, Style
from tqdm import tqdm
from pathlib import Path

from vasp_tools.check_OH import check_oh_dissociation
from vasp_tools.find_HH import find_hh_distances
from vasp_tools.reaction_time import save_reaction_times
from vasp_tools.track_hydrogen import track_molecular_hydrogen

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

##CONFIGURE##
# File paths
fp: Path = Path('/mnt/work_2/10diel-GGA-for-analysis/')
topology_file: str = str(fp.joinpath('10diel_20Li_64H2O/top.pdb'))

# Box size
box_size: float = 13.390

# Output reaction time file
output_file: str = '10_diel_reaction_times.csv'

# Trajectory files and folders
trajectory_files: Dict[int, Path] = {}
trajectory_folders: Dict[int, Path] = {}

# Global flag for stopping memory monitoring
stop_monitoring: bool = False

# Analysis type selection (choose what to compute)
ANALYSIS_TYPE: List[str] = ["reaction_times", "find_HH_distances", "check_OH_dissociation", "track_molecular_hydrogen"]
# Options: "reaction_times", "find_HH_distances", "check_OH_dissociation", "track_molecular_hydrogen"

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("reaction_processing.log", mode='w'), logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

# Choose the safest multiprocessing context
if sys.platform == "win32":
    ctx = mp.get_context("spawn")  # Windows needs "spawn"
else:
    ctx = mp.get_context("fork")  # Linux/macOS can use "fork"


def find_trajectories(root_folder: Path = fp) -> None:
    """
        Find all trajectories in the root folder.
    Args:
        root_folder (Path): The root folder to search for trajectories.

    Returns:
        None
    """

    find_folders = sorted(root_folder.glob("**/trj_*"))

    for folder in find_folders:
        try:
            file = folder.joinpath("traj.ALL.xyz")
            traj_number = int(folder.name.split('_')[-1])
            if file.exists():
                trajectory_folders[traj_number] = folder
                trajectory_files[traj_number] = file
            else:
                logger.info(f"Trajectory {traj_number} not found.")

        except Exception as e:
            print(e)


def save_pickle(data: Any, file_path: str) -> None:
    """
        Save data to a pickle file.
    Args:
        data (Any): Data to be saved.
        file_path (str): Path to the pickle file.

    Returns:
        None
    """

    try:
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        logger.error(f"Failed to save pickle file {file_path}: {e}")


def process_analysis(trajectory_number: int) -> None:
    """
        Process a single trajectory.
    Args:
        trajectory_number (int): The ID of the trajectory to process.

    Returns:
        None
    """

    folder_name: str = str(trajectory_folders[trajectory_number])
    trajectory_file: str = str(trajectory_files[trajectory_number])

    try:
        #  Get memory usage inside each worker
        pid = os.getpid()  # Current process ID

        # Load trajectory and set periodic box
        traj = md.load(trajectory_file, top=topology_file)
        traj.unitcell_lengths = np.full((traj.n_frames, 3), box_size / 10, dtype=np.float32)  # Convert Å to nm
        traj.unitcell_angles = np.full((traj.n_frames, 3), 90, dtype=np.float32)

        # Log memory usage
        mem_usage = psutil.Process(pid).memory_info().rss / (1024 * 1024)
        logger.info(f"Process {pid} (Trajectory {trajectory_number}) using {mem_usage:.2f} MB RAM")

        # Select all hydrogen atoms
        hs: np.ndarray = traj.topology.select('element H')
        Os: np.ndarray = traj.topology.select('element O')

        # Ensure unique H-H pairs
        h_pairs: np.ndarray = np.array([(h1, h2) for i, h1 in enumerate(hs) for h2 in hs[i + 1:]], dtype=int)

        # Selection of type of analysis
        for analysis in ANALYSIS_TYPE:
            result_file = f"{folder_name}/{analysis}.pkl"

            if analysis == "check_OH_dissociation":
                distances_oh_df = check_oh_dissociation(
                    traj=traj,
                    hs=hs,
                    os=Os,
                    file_path=f'{folder_name}'  # TODO: change to trajectory file
                )
                save_pickle(distances_oh_df, result_file)

            if analysis == "track_molecular_hydrogen":
                persistent_formations = track_molecular_hydrogen(
                    traj=traj,
                    h_pairs=h_pairs,
                    file_path=f'{folder_name}',  # TODO: change to trajectory file
                    output_indices=f'{folder_name}/indices.txt'
                )
                save_pickle(persistent_formations, result_file)

            if analysis == "find_HH_distances":
                distances_hh_df = find_hh_distances(
                    traj=traj,
                    h_pairs=h_pairs,
                    file_path=f'{folder_name}'  # TODO: change to trajectory file
                )
                save_pickle(distances_hh_df, result_file)

        del traj, hs, h_pairs, Os
        gc.collect()


    except Exception as e:
        logger.error(f"Trajectory {trajectory_number} failed: {e}")
        print(Fore.RED + f"[ERROR] Trajectory {trajectory_number} failed: {e}")
        print(Style.RESET_ALL)
        return None


def process_reaction_times(trajectory_number: int) -> Union[list[tuple[int, float]], list]:
    """
        Process a single trajectory reaction time.
    Args:
        trajectory_number (int): The ID of the trajectory to process.

    Returns:
        Union[list[tuple[int, float]], list]: List of reaction times.
    """

    trajectory_file: str = str(trajectory_files[trajectory_number])

    try:
        # Load trajectory and set periodic box
        traj = md.load(trajectory_file, top=topology_file)
        traj.unitcell_lengths = np.full((traj.n_frames, 3), box_size / 10, dtype=np.float32)  # Convert Å to nm
        traj.unitcell_angles = np.full((traj.n_frames, 3), 90, dtype=np.float32)

        # Select all hydrogen atoms
        hs: np.ndarray = traj.topology.select('element H')

        # Ensure unique H-H pairs
        h_pairs: np.ndarray = np.array([(h1, h2) for i, h1 in enumerate(hs) for h2 in hs[i + 1:]], dtype=int)
        reaction_times = save_reaction_times(traj=traj, h_pairs=h_pairs)

        del traj, hs, h_pairs
        gc.collect()

        return [(trajectory_number, time_i) for time_i in reaction_times]
    except Exception as e:
        logger.error(f"Trajectory {trajectory_number} failed: {e}")
        return []


def monitor_memory() -> None:
    """
        Monitor memory usage.
    Returns:
        None
    """

    global stop_monitoring
    while not stop_monitoring:
        total_memory = 0
        for process in psutil.process_iter(attrs=['pid', 'memory_info']):
            try:
                total_memory += process.info['memory_info'].rss  # Get memory usage (bytes)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue  # Ignore processes that disappear

        total_memory_mb = total_memory / (1024 * 1024)  # Convert bytes to MB
        logger.info(f"Total RAM Usage: {total_memory_mb:.2f} MB")

        sys.stdout.flush()  # Force live updates in some terminals
        time.sleep(10)  # Faster updates


def run_parallel_jobs(function: Any) -> None:
    """
        Runs a function in parallel using multiprocessing.
    Args:
        function (Any): check_OH_dissociation, find_hh_distances, track_molecular_hydrogen

    Returns:
        None
    """

    num_cores = max(mp.cpu_count() - 1, 1)
    parallel_jobs = min(10, num_cores)

    trajectory_jobs: List = list(trajectory_folders.keys())

    with ctx.Pool(processes=parallel_jobs) as pool:
        list(tqdm(pool.imap(function, trajectory_jobs),
                  total=len(trajectory_jobs),
                  desc=f"Processing trajectories",
                  ncols=100)
             )


def process_reaction_times_parallel() -> None:
    """
        Process reaction times in parallel.
    Returns:
        None
    """

    num_cores: int = max(mp.cpu_count() - 1, 1)  # Use at most 12 threads
    parallel_jobs: int = min(10, num_cores)  # Use 10 processes for balance

    trajectory_jobs: List = list(trajectory_folders.keys())

    with ctx.Pool(processes=parallel_jobs) as pool:
        results: List[List[Tuple[int, float]]] = list(
            tqdm(pool.imap(process_reaction_times, trajectory_jobs),
                 total=len(trajectory_jobs),
                 desc=f"Processing trajectories",
                 ncols=100)
        )

        #  Flatten results (list of lists → single list of tuples)
        all_reactions: List[Tuple[int, float]] = [entry for sublist in results for entry in sublist]

        #  Group by trajectory ID using default-dict
        reaction_dict = defaultdict(list)
        for traj_id, reaction_time in all_reactions:
            reaction_dict[traj_id].append(reaction_time)

        #  Sort and transpose results
        sorted_keys = sorted(reaction_dict.keys())
        max_length = max(len(times) for times in reaction_dict.values())
        transposed_data = [
            [reaction_dict[traj_id][i] if i < len(reaction_dict[traj_id]) else "" for traj_id in sorted_keys] for i
            in range(max_length)]

        #  Write transposed data to CSV
        with open(output_file, 'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(sorted_keys)  # Write headers (trajectory IDs)
            writer.writerows(transposed_data)  # Write transposed reaction times


def main() -> None:
    """
        Main function for analysis.
    Returns:
        None
    """

    global stop_monitoring

    find_trajectories(root_folder=fp)

    if "reaction_times" in ANALYSIS_TYPE:
        # Run reaction times in parallel
        process_reaction_times_parallel()

    if "find_HH_distances" in ANALYSIS_TYPE or "check_OH_dissociation" in ANALYSIS_TYPE or "track_molecular_hydrogen" in ANALYSIS_TYPE:
        # Run all other analyses in parallel
        run_parallel_jobs(process_analysis)

    stop_monitoring = True  # Stop memory monitoring


if __name__ == "__main__":
    try:
        stop_monitoring = False

        #  Start RAM monitoring thread before starting computation
        monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
        monitor_thread.start()

        #  Run main computation
        main()

        monitor_thread.join()  # Wait for monitoring thread to exit

        print(Fore.GREEN + "[INFO] Computation completed.")
        logger.info("Computation completed.")
        print(Style.RESET_ALL)

    except KeyboardInterrupt:
        print(Fore.RED + "[ERROR] Computation interrupted by user.")
        logger.error("Computation interrupted by user.")
        print(Style.RESET_ALL)

    except Exception as e:
        print(Fore.RED + f"[ERROR] An error occurred: {e}")
        logger.error(f"An error occurred: {e}")
        print(Style.RESET_ALL)
