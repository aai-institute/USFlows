import onnx
import os
import sys
import onnxruntime as ort
import numpy

from datetime import datetime
from src.explib.base import Experiment
from colorama import Fore


class OnnxConverter(Experiment):

    def __init__(self, path_flow: str, path_classifier: str, *args, **kwargs,) -> None:
        """Initialize verification experiment.
        Args:
            path_flow (string): The path to the flow model used for the verification experiment in ONNX format.
            path_classifier (string): Full path to the classifier in ONNX format.
        """
        super().__init__(*args, **kwargs)
        self.path_flow = path_flow
        self.path_classifier = path_classifier

    @staticmethod
    def dummy_verification(self, combined_model_path, maraboupy):
        options = maraboupy.Marabou.createOptions(verbosity=1)
        network = maraboupy.Marabou.read_onnx(combined_model_path)
        input_vars = network.inputVars[0]
        output_vars = network.outputVars[0][0]
        for i in range(len(input_vars)):
            network.setLowerBound(input_vars[i], -1)
            network.setUpperBound(input_vars[i], 1)

        var = maraboupy.MarabouPythonic.Var
        network.addConstraint(var(output_vars[0]) <= var(output_vars[1]))
        vals = network.solve(options=options)
        assignments = [vals[1][i] for i in range(100)]
        ort_sess_classifier = ort.InferenceSession(combined_model_path)
        outputs_classifier = ort_sess_classifier.run(
            None,
            {'x.1': numpy.asarray(assignments).astype(numpy.float32)})
        print(f' outputs of the classifier using onnxruntime: {outputs_classifier}')

    @staticmethod
    def swap_mul_inputs(self, model):
        # Ensures that the second input to the mul node is the constant node (and not the variable).
        # This constraint is imposed by the marabou library that implicitly
        # assumes the first input to be the variable and the second the constant factor.
        for node in model.graph.node:
            if node.op_type == "Mul":
                if "Constant" in node.input[0]:
                    node.input[0], node.input[1] = node.input[1], node.input[0]
        return model

    def fetch_directory(self, report_dir):
        current_time = str(datetime.now()).replace(" ", "")
        directory = f'{report_dir}/{self.name}/{current_time}'
        os.makedirs(directory)
        return directory

    @staticmethod
    def save_model(self, model, directory, model_name):
        saved_path = f'{directory}/{model_name}'
        onnx.save(model, f'{directory}/{model_name}')
        return saved_path

    @staticmethod
    def merge_models(self, modified_model, classifier):
        flow_output = modified_model.graph.output[0].name
        classifier_input = classifier.graph.input[0].name
        combined_model = onnx.compose.merge_models(
            modified_model, classifier,
            io_map=[(flow_output, classifier_input)]
        )
        onnx.checker.check_model(model=combined_model, full_check=True)
        return combined_model

    def conduct(self, report_dir: os.PathLike, storage_path: os.PathLike = None):
        model = onnx.load(self.path_flow)
        classifier = onnx.load(self.path_classifier)
        directory = self.fetch_directory(report_dir)
        self.save_model(model, directory, "unmodified_model.onnx")
        self.save_model(classifier, directory, "classifier.onnx")
        modified_model = self.swap_mul_inputs(model)
        combined_model = self.merge_models(modified_model,classifier)
        combined_model_path = self.save_model(combined_model, directory, "merged_model.onnx")

        try:
            sys.path.append('/home/mustafa/repos/Marabou')
            from maraboupy import Marabou
            import maraboupy
        except ImportError:
            Marabou = None
        if Marabou:
            print(f'Marabou available! {Marabou}')
            if combined_model:
                self.dummy_verification(combined_model_path, maraboupy)
        else:
            print(Fore.RED + 'Marabou not found!')
