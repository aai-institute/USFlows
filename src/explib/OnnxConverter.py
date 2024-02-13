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

    def dummy_verification(self, unmodified_flow, classifier, combined_model_path, marabou):
        options = marabou.createOptions(verbosity = 1)
        print(f'reading{combined_model_path}')
        network = marabou.read_onnx(combined_model_path)
        inputVars = network.inputVars[0]
        outputVars = network.outputVars[0][0]
        for i in range(len(inputVars)):
            network.setLowerBound(inputVars[i], -1)
            network.setUpperBound(inputVars[i], 1)
        # Patch inside of the MarabouPy library on commit master: 3374ed71 fix (#720)
        # is required due to the matmul layer there being considers as a single value multiplied with a vector,
        # but that in our case is rather a pairwise mul of the vector.
        # maraboupy / MarabouNetworkONNX.py
        # in line -1123,7 +1123,7 in class MarabouNetworkONNX(MarabouNetwork.MarabouNetwork):
        # e.addAddend(multiple[i], input1[i]) # changed from multiple to multiple[i]

        vals = network.solve(options=options)
        print(vals)
        assignments = [vals[1][i] for i in range(100)]
        ort_sess_classifier = ort.InferenceSession(combined_model_path)
        outputs_classifier = ort_sess_classifier.run(None, {'x.1': np.asarray(assignments).astype(np.float32)})
        print(outputs_classifier)
        vals[1]
    def swap_mul_inputs(self, model):
        for node in model.graph.node:
            if node.op_type == "Mul":
                if "Constant" in node.input[0]:
                    node.input[0], node.input[1] = node.input[1], node.input[0]
        return model

    def conduct(self, report_dir: os.PathLike, storage_path: os.PathLike = None):
        model = onnx.load(self.path_flow)
        classifier = onnx.load(self.path_classifier)

        # save input models into the reports target folder.
        # Might be redundant in the sense that the input models are most likely already saved elsewhere.
        # This way we can make sure that
        curtime = str(datetime.now()).replace(" ", "")
        dir = f'{report_dir}/{self.name}/{curtime}'
        combined_model_path = f'{dir}/merged_model.onnx'
        os.makedirs(dir)
        onnx.save(classifier, f'{dir}/classifier.onnx')
        onnx.save(model, f'{dir}/unmodified_model.onnx')

        modified_model = self.swap_mul_inputs(model)

        combined_model = None
        flow_output = model.graph.output[0].name
        classifier_input = classifier.graph.input[0].name
        combined_model = onnx.compose.merge_models(
            modified_model, classifier,
            io_map=[(flow_output, classifier_input)]
        )
        onnx.checker.check_model(model=combined_model, full_check=True)
        onnx.save(combined_model, combined_model_path)

        try:
            sys.path.append('/home/mustafa/repos/Marabou')
            from maraboupy import Marabou
        except ImportError:
            Marabou = None
        if Marabou:
            print(f'Marabou available! {Marabou}')
            if combined_model:
                self.dummy_verification(self.path_flow, self.path_classifier, combined_model_path, Marabou)
        else:
            print(Fore.RED + 'Marabou not found!')
