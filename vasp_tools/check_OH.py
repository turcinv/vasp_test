import mdtraj as md
import numpy as np
from typing import List, Tuple
import gc

def check_OH_dissociation(trajectory_file: str, topology_file: str, box_size: float, threshold: float = 2.0) -> None:
    """
    Analyzes a molecular dynamics trajectory to detect dissociation and reassociation of O-H bonds in water molecules
    based on O-H bonds exceeding a threshold distance.

    Parameters:
    - trajectory_file (str): Path to the trajectory file in XYZ format.
    - topology_file (str): Path to the topology file.
    - box_size (float): The simulation box size in Angstroms.
    - threshold (float, optional): Distance threshold in Angstroms for considering an O-H bond dissociated (default is 2.0 Å).

    Returns:
    - None: Prints dissociation events and the total number of dissociated bonds.
    """
    # Load trajectory and apply periodic box size
    traj: md.Trajectory = md.load(trajectory_file, top=topology_file)
    traj.unitcell_lengths = np.full((traj.n_frames, 3), box_size / 10, dtype=np.float32)  # Convert box size to nm
    traj.unitcell_angles = np.full((traj.n_frames, 3), 90, dtype=np.float32)

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

    for i, (h, o) in enumerate(OH_pairs):
        bond_label: str = f'Bond {i + 1} (H: {h}, O: {o})'
        dissociated_frames = np.where(distances[:, i] > threshold_nm)[0]

        if dissociated_frames.size > 0:
            dissociated_count += 1
            first_dissociation_time = round(float(dissociated_frames[0]) / 2000, 2)  # Assuming 2000 fps
            print(f"{bond_label} dissociated at {first_dissociation_time} ps")

    print(f"Total number of dissociated bonds: {dissociated_count}")

    # Clear memory
    del traj, topol, OH_pairs, distances, dissociated_frames, bond_label, first_dissociation_time
    gc.collect()
