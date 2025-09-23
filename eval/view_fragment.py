import argparse
from common_utils.myutils import DeserializeFragment
from PIL import Image
import io


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WebSocket fragment sender")

    parser.add_argument("fragment", help="the path to the fragment.")
    parser.add_argument(
        "-img",
        action="store_true",   # just a flag, no value required
        help="Display images contained in the fragment."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.fragment, "rb") as f:
        data: bytes = f.read()
        fragment: dict = DeserializeFragment(data)

        # loop over keys and print values
        for key, value in fragment.items():
            if key != 'images':
                print(f"{key}: {value}")

        # optionally open jpeg images
        if args.img:
            for idx, img_bytes in enumerate(fragment.get('images', [])):
                try:
                    img = Image.open(io.BytesIO(img_bytes))
                    img.show(title=f"Fragment Image {idx}")
                except Exception as e:
                    print(f"Error opening image {idx}: {e}")
