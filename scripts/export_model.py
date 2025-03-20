import os
from src.explib.config_parser import from_checkpoint

if __name__ == '__main__':
    print("Exporting model")
    PATHS = [
        "/home/mustafa/Documents/midas/power/_trial_27752_00001_1_batch_size=16,coupling_layers=2,coupling_nn_layers=5_5,nonlinearity=ref_ph_842f7f0d,lr=0.0002_2025-01-06_14-50-13/",
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
BUGFIX first: this script implicitly calls UntypedStorage::_load_from_bytes somewhere, which is missing the
param map_location.

torch/storage.py


def _load_from_bytes(b):
    return torch.load(io.BytesIO(b), map_location='cpu')

"""