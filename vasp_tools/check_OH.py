import mdtraj as md
import numpy as np
from typing import List, Tuple
import gc


def check_OH_dissociation(
        traj: md.Trajectory,
        hs: np.ndarray,
        os: np.ndarray,
        file_path: str,
        threshold: float = 2.0
) -> dict:
    """
    Analyzes a molecular dynamics trajectory to detect dissociation and reassociation of O-H bonds in water molecules
    based on O-H bonds exceeding a threshold distance.

    Parameters:
    - threshold (float, optional): Distance threshold in Angstroms for considering an O-H bond dissociated (default is 2.0 Å).

    Returns:
    - None: Prints dissociation events and the total number of dissociated bonds.
    """

    # Extract topology
    topol: md.Topology = traj.topology

    # Find O-H bonds dynamically from topology
    OH_pairs: List[Tuple[int, int]] = []

    for o in os:
        hydrogen_start_index = 2 * (o - os[0])
        OH_pairs.append((hs[hydrogen_start_index], o))
        OH_pairs.append((hs[hydrogen_start_index + 1], o))

    # # Compute O-H distances for all bonds in bulk
    # distances: np.ndarray = md.compute_distances(traj, OH_pairs, opt=True, periodic=True)  # nm

    # Check dissociation
    dissociated_count: int = 0
    threshold_nm: float = threshold / 10  # Convert Angstrom to nm

    with open(f'{file_path}/check_oh.log', "w", newline="") as f:
        dissociation = {}
        dissociated_count = 0
        for i, (h, o) in enumerate(OH_pairs):
            bond_label: str = f'Bond {i + 1} (H: {h}, O: {o})'
            distances = np.array(md.compute_distances(traj, [[h, o]], opt=True, periodic=True))[:, 0]
            dissociated = np.any(distances > (threshold / 10))  # Convert threshold to nm

            if dissociated:
                dissociated_count += 1
                first_dissociation_frame = np.argmax(distances > (threshold_nm))
                dissociation[i + 1] = {'H': h, 'O': o, 'first_dissociation_time': np.round(first_dissociation_frame/2000, 2)}
                print(f"{bond_label} dissociated at {np.round(first_dissociation_frame / 2000, 2)} ps", file=f, flush=True)


    # Clear memory
    del topol, OH_pairs, distances, bond_label
    gc.collect()

    return dissociation
