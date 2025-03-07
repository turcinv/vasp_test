import mdtraj as md
import numpy as np
from typing import List, Tuple
import gc


def check_OH_dissociation(
        traj: md.Trajectory,
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
    for bond in topol.bonds:
        if bond[0].element.symbol == 'O' and bond[1].element.symbol == 'H':
            OH_pairs.append((bond[1].index, bond[0].index))  # (H, O)
        elif bond[0].element.symbol == 'H' and bond[1].element.symbol == 'O':
            OH_pairs.append((bond[0].index, bond[1].index))  # (H, O)

    if not OH_pairs:
        raise ValueError("No O-H bonds detected in the topology.")

    # Compute O-H distances for all bonds in bulk
    distances: np.ndarray = md.compute_distances(traj, OH_pairs, opt=True, periodic=True)  # nm

    # Check dissociation
    dissociated_count: int = 0
    threshold_nm: float = threshold / 10  # Convert Angstrom to nm

    with open(f'{file_path}/check_oh.log', "w", newline="") as f:
        dissociation = {}
        for i, (h, o) in enumerate(OH_pairs):
            bond_label: str = f'Bond {i + 1} (H: {h}, O: {o})'
            dissociated_frames = np.where(distances[:, i] > threshold_nm)[0]

            if dissociated_frames.size > 0:
                dissociated_count += 1
                first_dissociation_time = round(float(dissociated_frames[0]) / 2000, 2)  # Assuming 2000 fps
                dissociation[i + 1] = {'H': h, 'O': o, 'first_dissociation_time': first_dissociation_time}
                print(f"{bond_label} dissociated at {first_dissociation_time} ps", file=f)

    # Clear memory
    del topol, OH_pairs, distances, dissociated_frames, bond_label, first_dissociation_time
    gc.collect()

    return dissociation
