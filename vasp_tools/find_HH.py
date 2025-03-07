import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
from typing import Optional, Dict
import gc

def find_HH_distances(
        traj: md.Trajectory,
        H_pairs: np.ndarray,
        file_path: str,
        threshold: float = 1.2,
        plot: bool = True
) -> Optional[pd.DataFrame]:
    """
    Analyzes a molecular dynamics trajectory to monitor all H-H distances
    that indicate potential molecular hydrogen formation.

    Parameters:
    - threshold (float): Distance threshold in Angstroms for potential bonding (default is 1.2 Å).
    - plot (bool): If True, plots the H-H distances; if False, returns the DataFrame.

    Returns:
    - If `plot=False`, returns a Pandas DataFrame containing H-H distances.
    - If `plot=True`, displays a plot and returns None.
    """

    # Convert threshold to nanometers
    threshold_nm: float = threshold / 10

    # Compute all H-H distances
    all_distances: np.ndarray = md.compute_distances(traj, H_pairs, opt=True, periodic=True)

    # Filter pairs that ever fall below the threshold
    bonded_indices: np.ndarray = np.where(all_distances.min(axis=0) < threshold_nm)[0]
    bonded_pairs: np.ndarray = H_pairs[bonded_indices]

    with open(f'{file_path}/HH_dist.log', 'w') as f:
        print(f"{len(bonded_pairs)} potential molecular hydrogen pairs detected.", file=f)

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
        plt.savefig(f'{file_path}/HH_dist.png', dpi=300)
        # plt.show()

        return None  # Explicitly returning None when plotting

    # Clear memory
    del H_pairs, all_distances, bonded_indices, bonded_pairs, distance_dict
    gc.collect()

    return distances_df  # Return DataFrame when not plotting
