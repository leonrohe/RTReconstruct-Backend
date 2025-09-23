import asyncio
import websockets
import argparse
import pathlib
import json
import struct

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WebSocket fragment sender")

    parser.add_argument("ws", help="The WebSocket server URI (e.g. ws://localhost:8765)")
    parser.add_argument("dir", help="The directory where the fragments are stored")
    parser.add_argument("--role", default="eval", help="Role to send in the handshake")
    parser.add_argument("--scene", default="eval", help="Scene to send in the handshake")
    parser.add_argument("--model", default="", help="If non empty, patches given model name into fragment")

    return parser.parse_args()

def patch_fragment(data: bytes, model: str) -> bytes:
    model_name_bytes = model.encode('utf-8')
    model_name_length_bytes = struct.pack('<I', len(model_name_bytes))
    splice = (model_name_length_bytes + model_name_bytes)
    
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
    
    magic = read_bytes(4)
    assert magic == b'LEON', f"Invalid magic bytes: {magic}"

    version = read_uint32()
    assert version == 1, f"Unsupported version: {version}"

    window = read_uint32()

    model_start = offset

    model_name_length = read_uint32()
    model_name = read_bytes(model_name_length).decode('utf-8')

    model_end = offset

    # put back together
    return data[:model_start] + splice + data[model_end:]


async def handshake(ws: websockets.WebSocketClientProtocol, role: str, scene: str) -> None:
    """Send handshake JSON with role and scene, and wait for server response."""
    msg = {"role": role, "scene": scene}
    await ws.send(json.dumps(msg))
    print(f"[handshake] Sent: {msg}")

    try:
        response = await ws.recv()
        print(f"[handshake] Server replied: {response}")
    except Exception as e:
        print(f"[handshake] Error during handshake: {e}")
        await ws.close(code=1008)
        raise

async def send_fragments(ws: websockets.WebSocketClientProtocol, directory: str, model: str) -> None:
    """Send all files in the given directory over the WebSocket, sorted by numeric index after the last underscore."""
    path = pathlib.Path(directory)

    if not path.exists() or not path.is_dir():
        raise ValueError(f"Invalid directory: {directory}")

    # collect files first
    files = [f for f in path.iterdir() if f.is_file()]

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

    files.sort(key=extract_key)

    for file in files:
        print(f"[send] Sending {file.name}...")
        data = file.read_bytes()

        if model:
            data = patch_fragment(data, model)

        await ws.send(data)

async def connect(uri: str, directory: str, role: str, scene: str, model: str) -> None:
    """Connect to the WebSocket and handle communication."""
    async with websockets.connect(uri) as ws:
        print(f"[connect] Connected to {uri}")
        await handshake(ws, role, scene)
        await send_fragments(ws, directory, model)
        print("[connect] All fragments sent.")

#  python .\eval\eval_client.py ws://192.168.178.28:5000/ws/client .\main\fragments\ --model neucon
if __name__ == "__main__":
    args = parse_args()
    asyncio.run(connect(args.ws, args.dir, args.role, args.scene, args.model))
