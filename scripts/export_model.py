import os
from src.explib.config_parser import from_checkpoint

if __name__ == '__main__':
    print("Exporting model")
    PATHS = [
        "/home/mustafa/Documents/midas/all_digits/mnist_ablation/0_mnist_basedist_comparison/mnist_basedist_comparison/0_mnist_3_laplace/",
        "/home/mustafa/Documents/midas/all_digits/mnist_ablation/1_mnist_basedist_comparison/mnist_basedist_comparison/0_mnist_3_laplace/",
        "/home/mustafa/Documents/midas/all_digits/mnist_ablation/2_mnist_basedist_comparison/mnist_basedist_comparison/0_mnist_3_laplace/",
        "/home/mustafa/Documents/midas/all_digits/mnist_ablation/3_mnist_basedist_comparison/mnist_basedist_comparison/0_mnist_3_laplace/",
        "/home/mustafa/Documents/midas/all_digits/mnist_ablation/4_mnist_basedist_comparison/mnist_basedist_comparison/0_mnist_3_laplace/",
        "/home/mustafa/Documents/midas/all_digits/mnist_ablation/5_mnist_basedist_comparison/mnist_basedist_comparison/0_mnist_3_laplace/",
        "/home/mustafa/Documents/midas/all_digits/mnist_ablation/6_mnist_basedist_comparison/mnist_basedist_comparison/0_mnist_3_laplace/",
        "/home/mustafa/Documents/midas/all_digits/mnist_ablation/7_mnist_basedist_comparison/mnist_basedist_comparison/0_mnist_3_laplace/",
        "/home/mustafa/Documents/midas/all_digits/mnist_ablation/8_mnist_basedist_comparison/mnist_basedist_comparison/0_mnist_3_laplace/",
        "/home/mustafa/Documents/midas/all_digits/mnist_ablation/digit9/experiment/"
    ]
    for PATH in PATHS:
        pkl = [f for f in os.listdir(PATH) if f.endswith("pkl")]
        print(f'picking the one: {pkl}')
        pkl = pkl[0]
        pts = [f for f in os.listdir(PATH) if f.endswith("pt")]
        print(f'picking the one: {pts}')
        pt = pts[0]
        model = from_checkpoint(PATH+pkl, PATH+pt)
        model = model.simplify()
        model.to_onnx(path=PATH+"forward.onnx", export_mode="forward")
        model.to_onnx(path=PATH+"backward.onnx", export_mode="backward")


"""
BUGFIX first: this script implicitly calls UntypedStorage::_load_from_bytes somehwere, which is missing the
param map_location.

torch/storage.py


def _load_from_bytes(b):
    return torch.load(io.BytesIO(b), map_location='cpu')

"""