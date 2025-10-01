import pathlib
import json
from typing import List
import argparse
from common_utils.myutils import DeserializeFragment

class Pose:
    def __init__(self, px, py, pz, rx, ry, rz, rw):
        self.px = px
        self.py = py
        self.pz = pz

        self.rx = rx
        self.ry = ry
        self.rz = rz
        self.rw = rw
    
    def __repr__(self):
        return f"Position: ({self.px}, {self.py}, {self.pz}), Rotation: ({self.rx}, {self.ry}, {self.rz}, {self.rw})"
    
    def to_dict(self):
        return {
            "position": {"x": self.px, "y": self.py, "z": self.pz},
            "rotation": {"x": self.rx, "y": self.ry, "z": self.rz, "w": self.rw},
        }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fragment replay pose extractor.")

    parser.add_argument("folder", help="The folder containing the numbered fragments.")

    return parser.parse_args()

def extract_poses_from_fragment(data: bytes) -> List[Pose]:
    fragment: dict = DeserializeFragment(data)

    poses = []
    for pose_dict in fragment['extrinsics']:
        pose: List = pose_dict['camera_position']
        rotation: List = pose_dict['camera_rotation']
        pose: Pose = Pose(*pose, *rotation)
        poses.append(pose)
    return poses

def extract_all_poses(folder: str) -> List[Pose]:
    def extract_key(f: pathlib.Path):
        # Use stem to ignore extensions (if any), and rpartition to split at last underscore
        stem = f.stem
        prefix, sep, idx_part = stem.rpartition('_')
        if not sep:
            # no underscore -> sort by name and put index -1 so pure-named files appear first
            return (stem, -1)
        try:
            idx = int(idx_part)
            return (prefix, idx)
        except ValueError:
            # non-integer suffix -> sort by prefix then by the raw suffix string
            return (prefix, idx_part)
    
    path = pathlib.Path(folder)

    if not path.exists() or not path.is_dir():
        raise ValueError(f"Invalid directory: {folder}")
    
    files = [f for f in path.iterdir() if f.is_file()]
    files.sort(key=extract_key)

    poses = []
    for i, file in enumerate(files):
        print(f"[INFO] Extracting file {i+1} out of {len(files)}")

        data: bytes = file.read_bytes()
        fragment_poses: List[Pose] = extract_poses_from_fragment(data)

        poses.extend(fragment_poses)

    return poses

if __name__ == "__main__":
    args = parse_args()
    poses = extract_all_poses(args.folder)

    # Convert all Pose objects to dicts
    poses_dicts = [pose.to_dict() for pose in poses]

    # Write to JSON file
    output_file = pathlib.Path(args.folder) / "poses.json"
    with open(output_file, "w") as f:
        json.dump(poses_dicts, f, indent=4)

    print(f"[INFO] Saved {len(poses)} poses to {output_file}")