from PIL import Image
import os
from glob import glob
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="resize images")

    parser.add_argument(
        "input",
        type=str,
        help="input images glob()"
    )
  
    parser.add_argument(
        "--x",
        type=int,
        default=640,
        help="new x size"
    )

    parser.add_argument(
        "--y",
        type=int,
        default = 480,
        help="new y size"
    )

    parser.add_argument(
        "output_folder",
        type=str,
        help="output folder"
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_folder): 
        os.makedirs(args.output_folder)

    for a in glob(args.input):
        img = Image.open( a )
        new_img = img.resize((args.x,args.y))
        new_img.save(os.path.join(args.output_folder, os.path.basename(a)))


