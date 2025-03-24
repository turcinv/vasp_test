import mdtraj as md
import numpy as np
from typing import List, Tuple, Any
import gc

def track_molecular_hydrogen(
    traj: md.Trajectory,
    h_pairs: np.ndarray,
    file_path: str,
    threshold: float = 1.2,
    write_indices: bool = True,
    output_indices: str = 'indices.txt'
):
    """
    Tracks the formation and persistence of molecular hydrogen in a molecular dynamics simulation
    based on the H-H distance being below a certain threshold.
    Optionally writes the hydrogen atom indices forming molecular hydrogen in a text file for visualization.

    Parameters:
    - threshold (float): Distance threshold in Ångströms for H-H bonding (default is 1.2 Å).
    - write_indices (bool): If True, writes H-H pairs forming persistent molecular hydrogen to a file.
    - output_indices (str): Filename to save persistent molecular hydrogen indices.

    Returns:
    - List of tuples [(H-H pair (h1, h2), first frame)] where molecular hydrogen was persistently formed.
    """

    # Convert threshold to nanometers
    threshold_nm: float = threshold / 10

    # Compute all H-H distances
    all_distances: np.ndarray = np.array(md.compute_distances(traj, h_pairs, opt=True, periodic=True))

    with open(f'{file_path}/track_H2.log', 'w') as f:
        # Identify persistent hydrogen pairs
        persistent_formations = []
        for idx, pair_distances in enumerate(all_distances.T):  # Iterate over H-H pairs
            below_threshold_frames = np.where(pair_distances < threshold_nm)[0]

            # If any sequence exists where bond is stable until the end
            for start_idx in below_threshold_frames:
                if np.all(pair_distances[start_idx:] < threshold_nm):
                    persistent_formations.append((tuple(h_pairs[idx]), start_idx))
                    print(f"Hydrogen pair {h_pairs[idx]} formed persistently from {np.round(start_idx / 2000, 2)} ps", file=f, flush=True)
                    break  # Stop checking once the first persistent formation is found

        persistent_count = len(persistent_formations)
        print(f"Total persistent molecular hydrogen formations: {persistent_count}", file=f, flush=True)

    # Write indices to file if requested
    if write_indices and persistent_formations:
        with open(output_indices, 'w') as f:
            for pair, _ in persistent_formations:
                f.write(f"{pair[0]} {pair[1]}\n")  # Each pair on a separate line

    # print(f"Total persistent molecular hydrogen formations: {len(persistent_formations)}")

    # Clear memory
    del all_distances, below_threshold_frames, start_idx, pair_distances
    gc.collect()

    return persistent_formations
