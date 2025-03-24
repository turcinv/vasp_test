import gc
from typing import List

import mdtraj as md
import numpy as np


def save_reaction_times(
        traj: md.Trajectory,
        H_pairs: np.ndarray,
        threshold: float = 1.2
) -> List[float]:
    """
    Detects and saves the first reaction times (in picoseconds) where hydrogen pairs form
    and persist as molecular hydrogen in a molecular dynamics trajectory.

    Parameters:
    - threshold (float): Distance threshold in Ångströms for H-H bonding (default: 1.2 Å).

    Returns:
    - List[float]: A sorted list of reaction times in picoseconds.
    """

    # Convert threshold to nanometers
    threshold_nm: float = threshold / 10

    # Compute all H-H distances
    all_distances: np.array = np.array(md.compute_distances(traj, H_pairs, opt=True, periodic=True))[:, 0]

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
    del all_distances, stable_start_indices, start_idx, reaction_time_ps
    gc.collect()

    # Return sorted reaction times
    return sorted(reaction_times)
