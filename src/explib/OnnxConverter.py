import onnx
import os
import sys
import onnxruntime as ort
import numpy as np

from datetime import datetime
from src.explib.base import Experiment
from colorama import Fore

class OnnxConverter(Experiment):

    def __init__(
        self,
        path_flow: str,
        path_classifier: str,
        *args,
        **kwargs,
    ) -> None:
        """Initialize hyperparameter optimization experiment.

        Args:
            path_flow (string): Full path to the experiment folder containing the .pt and .pkl file.
            path_classifier (string): Full path to the classifier in onnx format.
        """
        super().__init__(*args, **kwargs)
        self.path_flow = path_flow
        self.path_classifier = path_classifier

    def dummy_verification(self, unmodified_flow, classifier, combined_model_path, maraboupy):
        options = maraboupy.Marabou.createOptions(verbosity = 1)
        print(f'reading{combined_model_path}')
        network = maraboupy.Marabou.read_onnx(combined_model_path)
        inputVars = network.inputVars[0]
        outputVars = network.outputVars[0][0]
        for i in range(len(inputVars)):
            network.setLowerBound(inputVars[i], -1)
            network.setUpperBound(inputVars[i], 1)

        for j in range(len(outputVars)):
            eq_verified = maraboupy.Marabou.MarabouCore.Equation(maraboupy.Marabou.MarabouCore.Equation.EQ)
            network.addConstraint(maraboupy.MarabouPythonic.Var(outputVars[0]) <= maraboupy.MarabouPythonic.Var(outputVars[1]))

        vals = network.solve(options=options)
        print(vals)
        assignments = [vals[1][i] for i in range(100)]
        ort_sess_classifier = ort.InferenceSession(combined_model_path)
        outputs_classifier = ort_sess_classifier.run(None, {'x.1': np.asarray(assignments).astype(np.float32)})
        print(outputs_classifier)

    def swap_mul_inputs(self, model):
        for node in model.graph.node:
            if node.op_type == "Mul":
                if "Constant" in node.input[0]:
                    node.input[0], node.input[1] = node.input[1], node.input[0]
        return model

    def fetch_directory(self, report_dir):
        curtime = str(datetime.now()).replace(" ", "")
        directory = f'{report_dir}/{self.name}/{curtime}'
        os.makedirs(directory)
        return directory

    def save_model(self, model, directory, modelname):
        saved_path = f'{directory}/{modelname}'
        onnx.save(model, f'{directory}/{modelname}')
        return saved_path

    def conduct(self, report_dir: os.PathLike, storage_path: os.PathLike = None):
        model = onnx.load(self.path_flow)
        classifier = onnx.load(self.path_classifier)
        directory = self.fetch_directory(report_dir)
        self.save_model(model, directory, "unmodified_model.onnx")
        self.save_model(classifier, directory, "classifier.onnx")

        modified_model = self.swap_mul_inputs(model)

        flow_output = model.graph.output[0].name
        classifier_input = classifier.graph.input[0].name
        combined_model = onnx.compose.merge_models(
            modified_model, classifier,
            io_map=[(flow_output, classifier_input)]
        )
        onnx.checker.check_model(model=combined_model, full_check=True)
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
                self.dummy_verification(self.path_flow, self.path_classifier, combined_model_path, maraboupy)
        else:
            print(Fore.RED + 'Marabou not found!')
