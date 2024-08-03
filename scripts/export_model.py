
from src.explib.config_parser import from_checkpoint

if __name__ == '__main__':
    print("Exporting model")
    model = from_checkpoint("./config.pkl","./config.pt")
    model = model.simplify()
    model.to_onnx(path="./forward.onnx", export_mode="forward")
    model.to_onnx(path="./backward.onnx", export_mode="backward")
