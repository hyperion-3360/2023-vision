import argparse
import onnx_graphsurgeon as gs
import onnx
import numpy as np

def convert_to_float(input, output):
    try:
        graph = gs.import_onnx(onnx.load(input))
    except Exception as e:
        print(e)

    if graph:
        del_node = [node for node in graph.nodes if node.name == "Cast_0"][0]
        next_node = del_node.o()
        next_node.inputs = graph.inputs
        del_node.outputs.clear()
        graph.cleanup()

        for inp in graph.inputs:
            inp.dtype = np.float32
    try:
        onnx.save(gs.export_onnx(graph), output)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert_2_float")

    parser.add_argument(
        "input_onnx",
        type=str,
        help="ONNX model path",
    )

    parser.add_argument(
        "output_onnx",
        type=str,
        help="converted ONNX model path",
    )

    args = parser.parse_args()

    convert_to_float(args.input_onnx, args.output_onnx)