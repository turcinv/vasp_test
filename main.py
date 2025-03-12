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
from typing import List, Tuple, Any

import mdtraj as md
import numpy as np
import psutil
from colorama import Fore
from tqdm import tqdm

from vasp_tools.check_OH import check_OH_dissociation
from vasp_tools.find_HH import find_HH_distances
from vasp_tools.reaction_time import save_reaction_times
from vasp_tools.track_hydrogen import track_molecular_hydrogen

# DEBUG
DEBUG = False
if DEBUG:
    import random

# File paths
# fp: str = "/data/work/Water_reactivity/prod-GGA-vasp/"
fp: str ='/mnt/work_2/10diel-GGA-for-analysis/'
# topology_file: str = fp + '10diel_20Li_64H2O/top.pdb'
topology_file: str = fp + 'top.pdb'
box_size: float = 13.390
output_file: str = '10_diel_reaction_times.csv'

# Global flag for stopping memory monitoring
stop_monitoring: bool = False
batch_size: int = 100

# Analysis type selection (choose what to compute)
ANALYSIS_TYPE: List[str] = ["reaction_times", "find_HH_distances", "check_OH_dissociation", "track_molecular_hydrogen"]
# Options: "reaction_times", "find_HH_distances", "check_OH_dissociation", "track_molecular_hydrogen"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("reaction_processing.log", mode='w'), logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)


# --- GHOST ANALYSIS FUNCTIONS ---
def ghost_find_hh_distances():
    logger.info("Starting ghost analysis: Finding H-H distances")

    for i in range(3):
        time.sleep(0.5)
        hh_distance = random.uniform(0.5, 2.5)
        logger.debug(f"Simulated H-H distance: {hh_distance:.3f} Å")

        if hh_distance < 0.7:
            logger.warning("Detected possible hydrogen bond!")
        elif hh_distance > 2.0:
            logger.error("H-H distance too large! Possible broken structure.")

    logger.info("Ghost analysis complete: H-H distances checked.")


def ghost_find_oh_distances():
    logger.info("Starting ghost analysis: Checking OH dissociation")

    for i in range(3):
        time.sleep(0.5)
        dissociation_chance = random.random()
        logger.debug(f"Simulated dissociation probability: {dissociation_chance:.2%}")

        if dissociation_chance > 0.85:
            logger.warning("High dissociation probability detected!")
        elif dissociation_chance < 0.05:
            logger.info("Dissociation is unlikely under these conditions.")

    logger.info("Ghost analysis complete: OH dissociation checked.")


def ghost_track_molecular_hydrogen():
    logger.info("Starting ghost analysis: Tracking hydrogen atoms")

    for i in range(3):
        time.sleep(0.5)
        movement_vector = (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
        logger.debug(f"Simulated hydrogen movement vector: {movement_vector}")

        if sum(abs(x) for x in movement_vector) > 2.0:
            logger.warning("Unusually large hydrogen displacement detected!")
        elif sum(abs(x) for x in movement_vector) < 0.1:
            logger.error("Hydrogen appears stuck! Possible simulation issue.")

    logger.info("Ghost analysis complete: Hydrogen tracking finished.")


# Choose the safest multiprocessing context
if sys.platform == "win32":
    ctx = mp.get_context("spawn")  # Windows needs "spawn"
else:
    ctx = mp.get_context("fork")  # Linux/macOS can use "fork"


def save_pickle(data: Any, file_path: str):
    """Save data to a pickle file."""
    try:
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        logger.error(f"Failed to save pickle file {file_path}: {e}")


def process_analysis(i: int):
    """Processes a single trajectory and returns reaction times."""
    folder_name: str = f"trj_{i:03}"
    trajectory_file: str = f'{fp}10diel_20Li_64H2O/{folder_name}/traj.ALL.xyz'

    try:
        #  Get memory usage inside each worker
        pid = os.getpid()  # Current process ID

        # Load trajectory and set periodic box
        traj = md.load(trajectory_file, top=topology_file)
        traj.unitcell_lengths = np.full((traj.n_frames, 3), box_size / 10, dtype=np.float32)  # Convert Å to nm
        traj.unitcell_angles = np.full((traj.n_frames, 3), 90, dtype=np.float32)

        # Log memory usage
        mem_usage = psutil.Process(pid).memory_info().rss / (1024 * 1024)
        logger.info(f"Process {pid} (Trajectory {i}) using {mem_usage:.2f} MB RAM")

        # Select all hydrogen atoms
        hs: np.ndarray = traj.topology.select('element H')
        Os: np.ndarray = traj.topology.select('element O')

        # Ensure unique H-H pairs
        h_pairs: np.ndarray = np.array([(h1, h2) for i, h1 in enumerate(hs) for h2 in hs[i + 1:]], dtype=int)

        # Selection of type of analysis
        for analysis in ANALYSIS_TYPE:
            result_file = f"{fp}10diel_20Li_64H2O/{folder_name}/{analysis}.pkl"

            if analysis == "check_OH_dissociation":
                if DEBUG:
                    ghost_find_oh_distances()
                else:
                    distances_oh_df = check_OH_dissociation(
                        traj=traj,
                        hs=hs,
                        os=Os,
                        file_path=f'{fp}10diel_20Li_64H2O/{folder_name}'
                    )
                    save_pickle(distances_oh_df, result_file)

            if analysis == "track_molecular_hydrogen":
                if DEBUG:
                    ghost_track_molecular_hydrogen()
                else:
                    persistent_formations = track_molecular_hydrogen(
                        traj=traj,
                        H_pairs=h_pairs,
                        file_path=f'{fp}10diel_20Li_64H2O/{folder_name}',
                        output_indices=f'{fp}/10diel_20Li_64H2O/{folder_name}/indices.txt'
                    )
                    save_pickle(persistent_formations, result_file)

            if analysis == "find_HH_distances":
                if DEBUG:
                    ghost_find_hh_distances()
                else:
                    distances_hh_df = find_HH_distances(
                        traj=traj,
                        H_pairs=h_pairs,
                        file_path=f'{fp}10diel_20Li_64H2O/{folder_name}'
                    )
                    save_pickle(distances_hh_df, result_file)

        del traj, hs, h_pairs, os
        gc.collect()


    except Exception as e:
        logger.error(f"Trajectory {i} failed: {e}")
        print(Fore.RED + f"[ERROR] Trajectory {i} failed: {e}")
        return []


def process_reaction_times(i: int):
    """Processes reaction times and returns results."""
    folder_name: str = f"trj_{i:03}"
    trajectory_file: str = f'{fp}10diel_20Li_64H2O/{folder_name}/traj.ALL.xyz'

    try:
        # Load trajectory and set periodic box
        traj = md.load(trajectory_file, top=topology_file)
        traj.unitcell_lengths = np.full((traj.n_frames, 3), box_size / 10, dtype=np.float32)  # Convert Å to nm
        traj.unitcell_angles = np.full((traj.n_frames, 3), 90, dtype=np.float32)

        # Select all hydrogen atoms
        hs: np.ndarray = traj.topology.select('element H')

        # Ensure unique H-H pairs
        h_pairs: np.ndarray = np.array([(h1, h2) for i, h1 in enumerate(hs) for h2 in hs[i + 1:]], dtype=int)
        reaction_times = save_reaction_times(traj=traj, H_pairs=h_pairs)

        del traj, hs, h_pairs
        gc.collect()

        return [(i, time_i) for time_i in reaction_times]
    except Exception as e:
        logger.error(f"Trajectory {i} failed: {e}")
        return []


def monitor_memory():
    """Monitors and prints the current RAM usage every second, including child processes."""
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


def run_parallel_jobs(function, start: int, end: int):
    """Runs a function in parallel using multiprocessing."""
    num_cores = max(mp.cpu_count() - 1, 1)
    parallel_jobs = min(10, num_cores)

    with ctx.Pool(processes=parallel_jobs) as pool:
        list(tqdm(pool.imap(function, range(start, end)), total=(end - start),
                  desc=f"Processing {start}-{end}", ncols=100))


def process_reaction_times_batch(start: int, end: int):
    """Processes a batch of trajectories and writes grouped reaction times."""
    num_cores: int = max(mp.cpu_count() - 1, 1)  # Use at most 12 threads
    parallel_jobs: int = min(10, num_cores)  # Use 10 processes for balance

    with ctx.Pool(processes=parallel_jobs) as pool:
        results: List[List[Tuple[int, float]]] = list(
            tqdm(pool.imap(process_analysis, range(start, end)), total=(end - start),
                 desc=f"Processing {start}-{end}", ncols=100)
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
    """Main function to process all 100 trajectories in batches."""
    global stop_monitoring

    #  Process 100 trajectories in batches
    for batch_start in range(1, 101, batch_size):
        batch_end = min(batch_start + batch_size, 101)  # Avoid overshooting 100

        if "reaction_times" in ANALYSIS_TYPE:
            # Run reaction times in parallel
            process_reaction_times_batch(batch_start, batch_end)

        # Run all other analyses in parallel
        run_parallel_jobs(process_analysis, batch_start, batch_end)

    stop_monitoring = True  # Stop memory monitoring


if __name__ == '__main__':
    stop_monitoring = False

    #  Start RAM monitoring thread before starting computation
    monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
    monitor_thread.start()

    #  Run main computation
    main()

    monitor_thread.join()  # Wait for monitoring thread to exit

    print(Fore.GREEN + "[INFO] Computation completed.")
    logger.info("Computation completed.")
