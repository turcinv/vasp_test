import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
from typing import Optional, Dict
import gc

def find_HH_distances(
        trajectory_file: str,
        topology_file: str,
        box_size: float,
        threshold: float = 1.2,
        plot: bool = True
) -> Optional[pd.DataFrame]:
    """
    Analyzes a molecular dynamics trajectory to monitor all H-H distances
    that indicate potential molecular hydrogen formation.

    Parameters:
    - trajectory_file (str): Path to the trajectory file in XYZ format.
    - topology_file (str): Path to the topology file.
    - box_size (float): The simulation box size in Ångströms.
    - threshold (float): Distance threshold in Angstroms for potential bonding (default is 1.2 Å).
    - plot (bool): If True, plots the H-H distances; if False, returns the DataFrame.

    Returns:
    - If `plot=False`, returns a Pandas DataFrame containing H-H distances.
    - If `plot=True`, displays a plot and returns None.
    """

    # Load trajectory
    traj: md.Trajectory = md.load(trajectory_file, top=topology_file)
    traj.unitcell_lengths = np.full((traj.n_frames, 3), box_size / 10, dtype=np.float32)  # Convert to nm
    traj.unitcell_angles = np.full((traj.n_frames, 3), 90, dtype=np.float32)

    # Select hydrogen atoms and create valid pairs
    Hs: np.ndarray = traj.topology.select('element H')

    # Ensure unique H-H pairs
    H_pairs: np.ndarray = np.array([(h1, h2) for i, h1 in enumerate(Hs) for h2 in Hs[i + 1:]], dtype=int)

    # Convert threshold to nanometers
    threshold_nm: float = threshold / 10

    # Compute all H-H distances
    all_distances: np.ndarray = md.compute_distances(traj, H_pairs, opt=True, periodic=True)

    # Filter pairs that ever fall below the threshold
    bonded_indices: np.ndarray = np.where(all_distances.min(axis=0) < threshold_nm)[0]
    bonded_pairs: np.ndarray = H_pairs[bonded_indices]

    print(f"{len(bonded_pairs)} potential molecular hydrogen pairs detected.")

    # Construct DataFrame
    distance_dict: Dict[str, np.ndarray] = {
        f'Pair {i + 1} (H{h1}-H{h2})': all_distances[:, idx]
        for i, (h1, h2), idx in zip(range(len(bonded_pairs)), bonded_pairs, bonded_indices)
    }

    distances_df: pd.DataFrame = pd.DataFrame(distance_dict)

    # Plot if required
    if plot and not distances_df.empty:
        fig, ax = plt.subplots(figsize=(9, 4))

        for col in distances_df.columns:
            ax.plot(distances_df.index / 2000, distances_df[col] * 10, label=col, lw=1)

        ax.set_xlabel('Time [ps]', fontsize=12)
        ax.set_ylabel('Distance [Å]', fontsize=12)
        ax.set_title('H-H Distances Over Time', fontsize=14)
        ax.set_ylim(0, distances_df.max().max() * 10 + 1)  # Dynamic Y-limit
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1), title='H-H Pairs', fontsize=8)
        ax.grid(True)
        plt.tight_layout()
        plt.show()

        return None  # Explicitly returning None when plotting

    # Clear memory
    del traj, Hs, H_pairs, all_distances, bonded_indices, bonded_pairs, distance_dict
    gc.collect()

    return distances_df  # Return DataFrame when not plotting
