import numpy as np
import mdtraj as md
from typing import List
import gc

def save_reaction_times(
    trajectory_file: str,
    topology_file: str,
    box_size: float,
    threshold: float = 1.2
) -> List[float]:
    """
    Detects and saves the first reaction times (in picoseconds) where hydrogen pairs form
    and persist as molecular hydrogen in a molecular dynamics trajectory.

    Parameters:
    - trajectory_file (str): Path to the trajectory file.
    - topology_file (str): Path to the topology file.
    - box_size (float): Size of the simulation box in Ångströms.
    - threshold (float): Distance threshold in Ångströms for H-H bonding (default: 1.2 Å).

    Returns:
    - List[float]: A sorted list of reaction times in picoseconds.
    """

    # Load trajectory and set periodic box
    traj = md.load(trajectory_file, top=topology_file)
    traj.unitcell_lengths = np.full((traj.n_frames, 3), box_size / 10, dtype=np.float32)  # Convert Å to nm
    traj.unitcell_angles = np.full((traj.n_frames, 3), 90, dtype=np.float32)

    # Select hydrogen atoms
    Hs: np.ndarray = traj.topology.select('element H')

    # Ensure unique H-H pairs
    H_pairs = traj.topology.select_pairs(Hs, Hs)

    # Convert threshold to nanometers
    threshold_nm: float = threshold / 10

    # Compute all H-H distances
    all_distances: np.ndarray = md.compute_distances(traj, H_pairs, opt=True, periodic=True)

    # Identify first reaction times
    reaction_times: List[float] = []
    for pair_distances in all_distances.T:  # Iterate over H-H pairs
        stable_start_indices = np.where(pair_distances < threshold_nm)[0]
        for start_idx in stable_start_indices:
            if (pair_distances[start_idx:] < threshold_nm).all():
                reaction_time_ps = np.round(start_idx / 2000, 2)
                reaction_times.append(reaction_time_ps)
                break  # Stop after the first persistent segment is found for this pair

    # Clear memory
    del traj, Hs, H_pairs, all_distances, stable_start_indices, start_idx, reaction_time_ps
    gc.collect()

    # Return sorted reaction times
    return sorted(reaction_times)
