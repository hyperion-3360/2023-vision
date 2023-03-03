import os
from glob import glob
import argparse
import ntpath
import pathlib
import imagesize

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="convert annotation from relative to absolute based on image size")
 
    parser.add_argument(
        "images",
        type=str,
        help="image glob"
    )
 
    parser.add_argument(
        "label_folder",
        type=str,
        help="labels folder"
    )

    parser.add_argument(
        "output_folder",
        type=str,
        help="output folder"
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_folder): 
        os.makedirs(args.output_folder)

    for a in glob(args.images):
        head, tail = ntpath.split(a)
        base_label_name = pathlib.Path(tail).stem+".txt"

        input_file = os.path.join(args.label_folder, base_label_name)

        if not os.path.isfile(a):
            continue

        if not os.path.exists(a):
            continue

        width, height = imagesize.get(a)

        output_file = os.path.join(args.output_folder, base_label_name)

        with open(input_file, "r") as f:
            with open(output_file, "w") as out:
                lines = f.readlines()
                for l in lines:
                    columns  = l.split()
                    columns[1:]  = [ float(x) for x in columns[1:]]
                    columns[1]  *= width
                    columns[3]  *= width
                    columns[2]  *= height
                    columns[4]  *= height

                    columns[1:]  = [ int(x) for x in columns[1:]]
                    columns  = [ str(x) for x in columns]
                    out.write(" ".join(columns))
                    out.write("\n")




