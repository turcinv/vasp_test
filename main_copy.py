import multiprocessing as mp
import psutil
import time
import threading
from tqdm import tqdm
from typing import List, Tuple
import sys
import os
from collections import defaultdict
import csv
from colorama import Fore
import logging

from vasp_tools.reaction_time import save_reaction_times
from vasp_tools.find_HH import find_HH_distances
from vasp_tools.check_OH import check_OH_dissociation
from vasp_tools.track_hydrogen import track_molecular_hydrogen

# File paths
fp: str = "/data/work/Water_reactivity/prod-GGA-vasp/"
topology_file: str = fp + '10diel_20Li_64H2O/top.pdb'
box_size: float = 13.390
output_file: str = '10_diel_reaction_times.csv'

# Global flag for stopping memory monitoring
stop_monitoring: bool = False
batch_size: int = 100


# Configure logging
log_file = "reaction_processing.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='w'),  # Log file
        logging.StreamHandler(sys.stderr)  # Errors to terminal
    ]
)
logger = logging.getLogger(__name__)

# Choose the safest multiprocessing context
if sys.platform == "win32":
    ctx = mp.get_context("spawn")  # Windows needs "spawn"
else:
    ctx = mp.get_context("fork")  # Linux/macOS can use "fork"


def process_reaction_times(i: int) -> List[Tuple[int, float]]:
    """Processes a single trajectory and returns reaction times."""
    folder_name: str = f"trj_{i:03}"
    trajectory_file: str = f'{fp}10diel_20Li_64H2O/{folder_name}/traj.ALL.xyz'

    try:
        #  Get memory usage inside each worker
        pid = os.getpid()  # Current process ID
        mem_usage = psutil.Process(pid).memory_info().rss / (1024 * 1024)
        logger.info(f"Process {pid} (Trajectory {i}) using {mem_usage:.2f} MB RAM")

        reaction_times: List[float] = save_reaction_times(trajectory_file, topology_file, box_size)
        return [(i, time) for time in reaction_times]

    except Exception as e:
        logger.error(f"Trajectory {i} failed: {e}")
        print(Fore.RED + f"[ERROR] Trajectory {i} failed: {e}")
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


def process_batch_reaction_times(start: int, end: int):
    """Processes a batch of trajectories and writes grouped reaction times."""
    num_cores: int = max(mp.cpu_count() - 1, 1)  # Use at most 12 threads
    parallel_jobs: int = min(10, num_cores)  # Use 10 processes for balance

    with ctx.Pool(processes=parallel_jobs) as pool:
        results: List[List[Tuple[int, float]]] = list(
            tqdm(pool.imap(process_reaction_times, range(start, end)), total=(end - start),
                 desc=f"Processing {start}-{end}", ncols=100)
        )

    #  Flatten results (list of lists → single list of tuples)
    all_reactions: List[Tuple[int, float]] = [entry for sublist in results for entry in sublist]

    #  Group by trajectory ID using defaultdict
    reaction_dict = defaultdict(list)
    for traj_id, reaction_time in all_reactions:
        reaction_dict[traj_id].append(reaction_time)

    #  Sort and transpose results
    sorted_keys = sorted(reaction_dict.keys())
    max_length = max(len(times) for times in reaction_dict.values())
    transposed_data = [
        [reaction_dict[traj_id][i] if i < len(reaction_dict[traj_id]) else "" for traj_id in sorted_keys] for i in
        range(max_length)]

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
        process_batch_reaction_times(batch_start, batch_end)

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