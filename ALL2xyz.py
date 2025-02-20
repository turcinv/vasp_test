import os
import multiprocessing

def read_poscar_traj(file_path):
    """Reads a POSCAR-style trajectory file and extracts atomic positions."""
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    # Extract header and atom information
    header = lines[:7]
    atom_types = header[5].split()
    atom_counts = list(map(int, header[6].split()))
    num_atoms = sum(atom_counts)
    
    # Extract atomic positions
    positions = []
    step = 0
    for i in range(7, len(lines), num_atoms + 1):
        step += 1
        pos_block = lines[i + 1:i + 1 + num_atoms]
        positions.append((step, pos_block))
    
    return atom_types, atom_counts, positions

def write_xyz(atom_types, atom_counts, positions, output_file):
    """Writes extracted atomic positions to an XYZ file."""
    with open(output_file, "w") as f:
        for step, pos_block in positions:
            f.write(f"{sum(atom_counts)}\n")
            f.write(f"Step {step}\n")
            f.writelines(pos_block)

def convert_traj_to_xyz(traj_file):
    """Convert a single POSCAR trajectory file to XYZ format."""
    xyz_file = f"{traj_file}.xyz"
    atom_types, atom_counts, positions = read_poscar_traj(traj_file)
    write_xyz(atom_types, atom_counts, positions, xyz_file)
    print(f"Conversion completed: {xyz_file}")

def main():
    """Parallelize the conversion of multiple POSCAR trajectories."""
    traj_files = [file for file in os.listdir(".") if file.startswith("traj") and file.endswith(".ALL")]
    
    if not traj_files:
        print("No traj.ALL files found!")
        return
    
    print(f"Found {len(traj_files)} trajectory files.")
    
    with multiprocessing.Pool(processes=min(len(traj_files), os.cpu_count())) as pool:
        pool.map(convert_traj_to_xyz, traj_files)
    
    print("All conversions completed.")

if __name__ == "__main__":
    main()
