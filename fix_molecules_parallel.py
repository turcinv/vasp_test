import os
import multiprocessing
import mdtraj as md
import numpy as np
from tqdm import tqdm
import gc
from typing import List, Tuple


def fix_molecules(top: str, traj: str, box_dimension: float) -> None:
    """Process a molecular dynamics trajectory to fix broken molecules across periodic boundaries using MDTraj.

    Args:
        top (str): Path to the topology file.
        traj (str): Path to the trajectory file.
        box_dimension (float): The size of the periodic box.
    """
    try:
        traj_data: md.Trajectory = md.load(traj, top=top)

        # ✅ Manually define periodic unit cell dimensions
        traj_data.unitcell_lengths = np.full((traj_data.n_frames, 3), box_dimension / 10, dtype=np.float32)  # nm
        traj_data.unitcell_angles = np.full((traj_data.n_frames, 3), 90, dtype=np.float32)  # Assume cubic box

        # ✅ Apply PBC correction
        traj_data.image_molecules(inplace=True)

        directory: str = os.path.dirname(traj)
        outfile: str = os.path.join(directory, f'{os.path.basename(traj)}.fixed_test.xyz')
        print(f"✅ Saving trajectory in file: {outfile}")

        with open(outfile, "w") as f:
            for frame in tqdm(traj_data, desc='Processing frames'):
                f.write(f"{frame.n_atoms}\n")
                f.write("Frame\n")
                for atom, pos in zip(frame.topology.atoms, frame.xyz[0]):
                    f.write(f"{atom.element.symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

        del traj_data, directory, outfile
        gc.collect()

    except Exception as e:
        print(f"❌ Error processing {traj}: {e}")


def process_trajectories(args: Tuple[str, str, float]) -> None:
    """Wrapper function for multiprocessing.

    Args:
        args (Tuple[str, str, float]): Tuple containing topology file path, trajectory file path, and box size.
    """
    top, traj, box_dimension = args
    fix_molecules(top, traj, box_dimension)


def main(topology_file: str, trajectory_files: List[str], box_dimension: float) -> None:
    """Parallelize processing of multiple trajectory files.

    Args:
        topology_file (str): Path to the topology file.
        trajectory_files (List[str]): List of trajectory file paths.
        box_dimension (float): The size of the periodic box.
    """
    args_list: List[Tuple[str, str, float]] = [(topology_file, traj, box_dimension) for traj in trajectory_files]

    with multiprocessing.Pool(processes=min(len(trajectory_files), os.cpu_count())) as pool:
        pool.map(process_trajectories, args_list)

    print("✅ All trajectory fixes completed.")


if __name__ == "__main__":
    topology_file: str = "../top-noI.pdb"  # Example topology file
    base_dir: str = "."  # Current directory
    trajectory_files: List[str] = []

    # Recursively search for trajectory files in subdirectories named trj_*
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path) and folder.startswith("trj_"):
            trajectory_files.extend([
                os.path.join(folder_path, file)
                for file in os.listdir(folder_path)
                if file.startswith("traj") and file.endswith(".ALL")
            ])

    if not trajectory_files:
        print("❌ No traj.ALL files found!")
        exit(1)

    print(f"✅ Found {len(trajectory_files)} trajectory files.")

    box_dimension: float = 12.530  # Example box size

    main(topology_file, trajectory_files, box_dimension)
