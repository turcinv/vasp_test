import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
from typing import Optional
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
    bonded_indices = bonded_indices.astype(int)



    if bonded_indices.size > 0:
        bonded_indices = bonded_indices.astype(int)  # Ensure it's integer-based
        valid_bonded_indices = bonded_indices[bonded_indices < H_pairs.shape[0]]  # Ensure no out-of-bounds
        bonded_pairs = H_pairs[valid_bonded_indices]
    else:
        bonded_pairs = np.empty((0, 2), dtype=int)  # Handle empty case

    # Identifying pairs that form and then remain formed
    persistent_formations = []
    persistent_count = 0

    with open(f'{file_path}/HH_dist.log', 'w') as f:
        for idx, pair_distances in enumerate(all_distances.T):  # Transpose to iterate over pairs
            stable_start_indices = np.where(pair_distances < threshold_nm)[0]

            # Check if there's any segment from some index till the end that's below threshold
            for start_idx in stable_start_indices:
                if (pair_distances[start_idx:] < threshold_nm).all():
                    persistent_formations.append((H_pairs[idx], start_idx))
                    persistent_count += 1
                    print(f"Hydrogen pair {H_pairs[idx]} formed molecular hydrogen persistently from {np.round(start_idx / 2000, 2)} ps",
                          file=f, flush=True)
                    break  # Stop after the first persistent segment is found for this pair

        print(f"Total persistent formations: {persistent_count}", file=f, flush=True)

    distance_dict = {f'Pair {i + 1} (H{pair[0]}-H{pair[1]})': all_distances[:, idx]
                     for i, pair in enumerate(bonded_pairs)
                     for idx in np.where((H_pairs == pair).all(axis=1))[0]}

    distances_df: pd.DataFrame = pd.DataFrame(distance_dict)

    # Plot if required
    if not plot and not distances_df.empty:
        print("Warning: distances_df is empty, skipping plot.")
        return None

    fig, ax = plt.subplots(figsize=(9, 4))
    time_index = np.arange(distances_df.shape[0]) / 2000  # Ensure valid time scale

    for col in distances_df.columns:
        try:
            distances_df[col] = distances_df[col].fillna(0)
            distances = np.array(distances_df[col], dtype=np.float64) * 10
            ax.plot(time_index, distances, label=col, lw=1)
        except Exception as e:
            print(f"Error while processing column {col}: {e}")

    ax.set_xlabel('Time [ps]', fontsize=12)
    ax.set_ylabel('Distance [Å]', fontsize=12)
    ax.set_title('H-H Distances Over Time', fontsize=14)
    ax.set_ylim(0.5, 6)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1), title='H-H Pairs', fontsize=8)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f'{file_path}/HH_dist.png', dpi=300)


    # Clear memory
    del H_pairs, all_distances, bonded_indices, bonded_pairs, distance_dict
    gc.collect()

    return distances_df  # Return DataFrame when not plotting
