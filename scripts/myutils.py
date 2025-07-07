import io
import struct

from PIL import Image
from typing import Any, Dict

import numpy as np

class ModelResult():
    def __init__(self, scene_name: str, output: bytes):
        self.scene_name = scene_name
        self.output = output
        self.version = 2

    def Serialize(self) -> bytes:
        """
        Serialize the model result to bytes.
        This method should be implemented by subclasses.
        """
        scene_name_bytes = self.scene_name.encode('utf-8')
        scene_name_length = len(scene_name_bytes)

        # Prepare the output with scene name length and scene name as single byte array
        result = (
            b'LEON',
            self.version.to_bytes(4, 'little'),
            scene_name_length.to_bytes(4, 'little') +
            scene_name_bytes +  # Scene name bytes
            self.output  # Actual output data
        )
        return result

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
    assert fragment_magic == b'LEON', f"Invalid magic bytes: {fragment_magic}"

    fragment_version = read_uint32()
    assert fragment_version == 1, f"Unsupported version: {fragment_version}"

    fragment_window = read_uint32()
    result['window_size'] = fragment_window

    model_name_length = read_uint32()
    model_name = read_bytes(model_name_length).decode('utf-8')
    result['model_name'] = model_name

    scene_name_length = read_uint32()
    scene_name = read_bytes(scene_name_length).decode('utf-8')
    result['scene_name'] = scene_name

    # --- Images ---
    images = []
    for _ in range(fragment_window):
        image_size = read_uint32()
        image_data = read_bytes(image_size)

        image_width = read_float()
        image_height = read_float()
        # image_dimensions = np.array([image_width, image_height], dtype=np.float32)

        images.append(image_data)
    result['images'] = images

    # --- Intrinsics ---
    intrinsics = []
    for _ in range(fragment_window):
        focal_length_x = read_float()
        focal_length_y = read_float()
        focal_length = np.array([focal_length_x, focal_length_y], dtype=float)

        principal_point_x = read_float()
        principal_point_y = read_float()
        principal_point = np.array([principal_point_x, principal_point_y], dtype=float)

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
        camera_position = np.array([camera_position_x, camera_position_y, camera_position_z], dtype=float)

        camera_rotation_x = read_float()
        camera_rotation_y = read_float()
        camera_rotation_z = read_float()
        camera_rotation_w = read_float()
        camera_rotation = np.array([camera_rotation_x, camera_rotation_y, camera_rotation_z, camera_rotation_w], dtype=float)

        extrinsics.append({
            'camera_position': camera_position,
            'camera_rotation': camera_rotation
        })
    result['extrinsics'] = extrinsics

    return result

def DeserializeResult(data: bytes) -> ModelResult:
    """
    Deserialize the model result from bytes.
    This function assumes the data is in the format defined by ModelResult.Serialize().
    """
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

    # Read magic bytes
    magic = read_bytes(4)
    assert magic == b'LEON', f"Invalid magic bytes: {magic}"

    # Read version
    version = read_uint32()
    assert version == 2, f"Unsupported version: {version}"

    # Read scene name length and scene name
    scene_name_length = read_uint32()
    scene_name = read_bytes(scene_name_length).decode('utf-8')

    # Read output data
    output = mv[offset:].tobytes()

    return ModelResult(scene_name, output)