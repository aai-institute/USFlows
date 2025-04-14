import os
from src.explib.config_parser import from_checkpoint

if __name__ == '__main__':
    print("Exporting model")
    PATHS = [
        "/home/mustafa.yalciner/_trial_2025-04-14_11-49-20/_trial_bd102_00000_0_batch_size=1024,nonlinearity=ref_ph_842f7f0d_2025-04-14_11-49-20/",
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