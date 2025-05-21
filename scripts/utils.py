import io
import struct

from PIL import Image
from typing import Any, Dict

import numpy as np


def DeserializeFragment(data: bytes) -> Dict[str, Any]:
    mv = memoryview(data)
    offset = 0

    def read_bytes(length: int) -> bytes:
        nonlocal offset
        val = mv[offset:offset+length]
        offset += length
        return val.tobytes()

    def read_uint32() -> int:
        nonlocal offset
        val = struct.unpack_from('<I', mv, offset)[0]
        offset += 4
        return val

    def read_float() -> float:
        nonlocal offset
        val = struct.unpack_from('<f', mv, offset)[0]
        offset += 4
        return val

    result: Dict[str, Any] = {}

    # --- Fragment Header ---
    fragment_magic = read_bytes(4)
    if fragment_magic != b'LEON':
        print(f"Invalid magic bytes: {fragment_magic}")
        return {}

    fragment_version = read_uint32()
    print(f"Fragment version: {fragment_version}")

    fragment_window = read_uint32()
    print(f"Fragment window size: {fragment_window}")

    model_name_length = read_uint32()
    model_name = read_bytes(model_name_length).decode('utf-8')
    print(f"Model name: {model_name}")

    # --- Images ---
    images = []
    for _ in range(fragment_window):
        image_size = read_uint32()
        image_data = read_bytes(image_size)

        image_width = read_float()
        image_height = read_float()
        image_dimensions = np.array([image_width, image_height], dtype=np.float32)

        images.append(Image.open(io.BytesIO(image_data)))
    result['images'] = images

    # --- Intrinsics ---
    intrinsics = []
    for _ in range(fragment_window):
        focal_length_x = read_float()
        focal_length_y = read_float()
        focal_length = np.array([focal_length_x, focal_length_y], dtype=np.float32)

        principal_point_x = read_float()
        principal_point_y = read_float()
        principal_point = np.array([principal_point_x, principal_point_y], dtype=np.float32)

        intrinsics.append({
            'focal_length': focal_length,
            'principal_point': principal_point
        })
    result['intrinsics'] = intrinsics

    # --- Extrinsics ---
    extrinsics = []
    for _ in range(fragment_window):
        camera_position_x = read_float()
        camera_position_y = read_float()
        camera_position_z = read_float()
        camera_position = np.array([camera_position_x, camera_position_y, camera_position_z], dtype=np.float32)

        camera_rotation_x = read_float()
        camera_rotation_y = read_float()
        camera_rotation_z = read_float()
        camera_rotation_w = read_float()
        camera_rotation = np.array([camera_rotation_x, camera_rotation_y, camera_rotation_z, camera_rotation_w], dtype=np.float32)

        extrinsics.append({
            'camera_position': camera_position,
            'camera_rotation': camera_rotation
        })
    result['extrinsics'] = extrinsics

    return result