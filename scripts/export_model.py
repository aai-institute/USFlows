
from src.explib.config_parser import from_checkpoint

if __name__ == '__main__':
    print("Exporting model")
    model = from_checkpoint("./a.pkl","./a.pt")
    model = model.simplify()
    model.to_onnx(path="./forward.onnx", export_mode="forward")
    model.to_onnx(path="./backward.onnx", export_mode="backward")


"""
BUGFIX first: this script implicitly calls UntypedStorage::_load_from_bytes somehwere, which is missing the
param map_location.

torch/storage.py


def _load_from_bytes(b):
    return torch.load(io.BytesIO(b), map_location='cpu')

"""