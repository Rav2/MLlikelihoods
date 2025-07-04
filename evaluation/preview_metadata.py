import os, sys
import onnx

def main(model_path):
    onnx_model = onnx.load(model_path)
    for em in sorted(onnx_model.metadata_props, key=lambda em: em.key):
        print(f'{em.key}\t:\t{em.value}')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('python preview_metadata.py <onnx-model-parh>')
        exit()
    else:
        main(sys.argv[1])