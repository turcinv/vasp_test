import os
import multiprocessing
import mdtraj as md
import numpy as np
from tqdm import tqdm
import gc

def fix_molecules(top, traj, box_dimension):
    """Process a molecular dynamics trajectory to fix the broken molecules across periodic boundaries using MDTraj."""
    traj_data = md.load(traj, top=top)
    box_dimension = float(box_dimension)
    
    # Apply PBC to make molecules whole
    traj_data.image_molecules(inplace=True)
    
    directory = os.path.dirname(traj)
    outfile = os.path.join(directory, f'{os.path.basename(traj)}.fixed.xyz')
    print(f"Saving trajectory in file: {outfile}")
    
    with open(outfile, "w") as f:
        for frame in tqdm(traj_data, desc='Processing frames'):
            f.write(f"{frame.n_atoms}\n")
            f.write("Frame\n")
            for atom, pos in zip(frame.topology.atoms, frame.xyz[0]):
                f.write(f"{atom.element.symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
                
    del traj_data, box_dimension, directory, outfile
    gc.collect()
    
    

def process_trajectories(args):
    """Wrapper function for multiprocessing."""
    top, traj, box_dimension = args
    fix_molecules(top, traj, box_dimension)

def main(topology_file, trajectory_files, box_dimension):
    """Parallelize processing of multiple trajectory files."""
    args_list = [(topology_file, traj, box_dimension) for traj in trajectory_files]
    
    with multiprocessing.Pool(processes=min(len(trajectory_files), os.cpu_count())) as pool:
        pool.map(process_trajectories, args_list)
    
    print("All trajectory fixes completed.")

if __name__ == "__main__":
    topology_file = "../top-noI.pdb"  # Example topology file
    trajectory_files = [file for file in os.listdir(".") if file.endswith(".xyz")]
    box_dimension = 13.390  # Example box size
    
    main(topology_file, trajectory_files, box_dimension)
