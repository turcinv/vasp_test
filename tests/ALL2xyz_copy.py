import os
import multiprocessing
from typing import List
from pymatgen.core import Structure

# Define lattice parameters for VASP box (update if needed)
LATTICE = [[12.530, 0, 0], [0, 12.530, 0], [0, 0, 12.530]]


def read_vasp_traj(file_path: str):
    """Reads a POSCAR-style trajectory file and extracts atomic positions as pymatgen Structures."""
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        header = lines[:7]  # First 7 lines contain POSCAR header
        atom_types = header[5].split()
        atom_counts = list(map(int, header[6].split()))
        num_atoms = sum(atom_counts)

        # Generate atom labels for each atom
        atom_labels = []
        for atom_type, count in zip(atom_types, atom_counts):
            atom_labels.extend([atom_type] * count)

        frames = []
        for i in range(7, len(lines), num_atoms + 1):
            pos_block = lines[i + 1:i + 1 + num_atoms]

            # If step is incomplete, skip it
            if len(pos_block) != num_atoms:
                print(f"⚠️ Skipping incomplete step at line {i + 1} in {file_path}")
                continue

            coords = [list(map(float, line.split())) for line in pos_block]
            structure = Structure(LATTICE, atom_labels, coords, coords_are_cartesian=True)
            frames.append(structure)

        print(f"✅ Loaded {len(frames)} frames from {file_path}")
        return frames

    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return []


def write_xyz(frames: List[Structure], output_file: str):
    """Writes pymatgen structures to an XYZ trajectory file."""
    try:
        with open(output_file, "w") as f:
            for i, structure in enumerate(frames):
                f.write(f"{len(structure.sites)}\n")
                f.write(f"Step {i + 1}\n")
                for site in structure.sites:
                    f.write(f"{site.species_string} {site.x:.6f} {site.y:.6f} {site.z:.6f}\n")
        print(f"✅ Converted to XYZ: {output_file}")
    except Exception as e:
        print(f"❌ Error writing XYZ: {e}")


def write_poscars(frames: List[Structure], output_dir: str):
    """Writes each trajectory frame as a separate POSCAR file."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        for i, structure in enumerate(frames):
            filename = os.path.join(output_dir, f"POSCAR_step_{i + 1:03d}")
            structure.to(fmt="poscar", filename=filename)
        print(f"✅ Saved {len(frames)} POSCAR files in {output_dir}")
    except Exception as e:
        print(f"❌ Error writing POSCARs: {e}")


def process_trajectory(file_path: str):
    """Process a single trajectory file: Read and save as XYZ/POSCAR."""
    frames = read_vasp_traj(file_path)
    if not frames:
        return

    # Define output filenames
    traj_dir = os.path.dirname(file_path)
    traj_name = os.path.basename(traj_dir)  # Extract folder name like "trj_001"
    xyz_output = os.path.join(traj_dir, f"{traj_name}.xyz")
    poscar_output_dir = os.path.join(traj_dir, f"{traj_name}_POSCARs")

    # Save outputs
    write_xyz(frames, xyz_output)
    write_poscars(frames, poscar_output_dir)


def main():
    """Find all traj.ALL files and process them in parallel."""
    base_dir = "."  # Current directory
    traj_files = []

    # Search for traj.ALL files in trj_* directories
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path) and folder.startswith("trj_"):
            traj_file = os.path.join(folder_path, "traj.ALL")
            if os.path.exists(traj_file):
                traj_files.append(traj_file)

    if not traj_files:
        print("❌ No traj.ALL files found!")
        return

    print(f"✅ Found {len(traj_files)} trajectory files.")

    # Parallel processing
    num_workers = min(len(traj_files), os.cpu_count() - 1)
    print(f"🚀 Using {num_workers} parallel workers.")

    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(process_trajectory, traj_files)

    print("✅ All conversions completed.")


if __name__ == "__main__":
    main()
