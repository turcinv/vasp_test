from pymatgen.core import Structure
import os

def read_vasp_traj(file_path):
    """Reads a POSCAR-style trajectory file and extracts atomic positions."""
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Extract header
    header = lines[:7]  # First 7 lines contain POSCAR header
    atom_types = header[5].split()
    atom_counts = list(map(int, header[6].split()))
    num_atoms = sum(atom_counts)

    # Create atom labels for each atom
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

        # Convert to Pymatgen Structure
        coords = [list(map(float, line.split())) for line in pos_block]
        structure = Structure(lattice=[[12.530, 0, 0], [0, 12.530, 0], [0, 0, 12.530]],
                              species=atom_labels, coords=coords, coords_are_cartesian=True)
        frames.append(structure)

    print(f"Loaded {len(frames)} frames from {file_path}")
    return frames

def write_xyz(frames, output_file):
    """Writes pymatgen structures to an XYZ trajectory file."""
    with open(output_file, "w") as f:
        for i, structure in enumerate(frames):
            f.write(f"{len(structure.sites)}\n")
            f.write(f"Step {i+1}\n")
            for site in structure.sites:
                f.write(f"{site.species_string} {site.x:.6f} {site.y:.6f} {site.z:.6f}\n")

    print(f"✅ Converted to XYZ: {output_file}")

# Convert traj.ALL to XYZ
input_traj = "traj.ALL"
output_xyz = "traj.xyz"

frames = read_vasp_traj(input_traj)
if frames:
    write_xyz(frames, output_xyz)


