from pathlib import Path
from typing import Dict

find_folders = sorted(Path("/mnt/work_2/10diel-GGA-for-analysis").glob("**/trj_*"))
trajectory_files: Dict[int, Path] = {}
trajectory_folders: Dict[int, Path] = {}

for folder in find_folders:
    try:
        file = folder.joinpath("traj.ALL.xyz")
        traj_number = int(folder.name.split('_')[-1])
        if file.exists():
            trajectory_folders[traj_number] = folder
            trajectory_files[traj_number] = file

    except Exception as e:
        print(e)

print(trajectory_folders[1])
