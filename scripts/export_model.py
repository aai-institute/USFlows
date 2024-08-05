
from src.explib.config_parser import from_checkpoint

if __name__ == '__main__':
    print("Exporting model")
    PATH = "/home/mustafa/Documents/midas/flows_unscaled_mnist/_trial_81c50_00004_4_batch_size=128,coupling_layers=2,coupling_nn_layers=300,lr=0.0001_2024-08-03_18-21-58/"
    pkl = "params.pkl"
    pt = "checkpoint.pt"
    model = from_checkpoint(PATH+pkl, PATH+pt)
    model = model.simplify()
    model.to_onnx(path=PATH+"/forward.onnx", export_mode="forward")
    model.to_onnx(path=PATH+"./backward.onnx", export_mode="backward")


"""
BUGFIX first: this script implicitly calls UntypedStorage::_load_from_bytes somehwere, which is missing the
param map_location.

torch/storage.py


def _load_from_bytes(b):
    return torch.load(io.BytesIO(b), map_location='cpu')

"""