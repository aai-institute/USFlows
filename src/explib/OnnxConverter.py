import click
import onnx
import onnxruntime as ort
import os
import sys
import torch

from datetime import datetime
from src.explib.base import Experiment
from colorama import Fore
from sam4onnx import modify

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

    def convert_to_IR_8(self, model):
        model.ir_version = 8
        return model

    def compare_models(self, model_unmodified_path, model_modified):
        ort_sess_approved = ort.InferenceSession(model_unmodified_path)
        ort_sess_modified = ort.InferenceSession(model_modified.SerializeToString())
        diff_counter = 0
        random_test_counts = 100
        for i in range(random_test_counts):
            x = torch.rand(100)
            outputs_approved = ort_sess_approved.run(None, {'x.1': x.numpy()})
            outputs_modified = ort_sess_modified.run(None, {'x.1': x.numpy()})
            for i in range(len(outputs_approved[0])):
                diff = abs(outputs_approved[0][i] - outputs_modified[0][i])
                if diff > 0.001:
                    diff_counter = diff_counter + 1
                    print(diff)
        print(f'diff counter: {diff_counter}')
        return diff_counter

    def fetch_weights(self, model, constant_name):
        for ini in model.graph.initializer:
            if ini.name in constant_name:
                arr = onnx.numpy_helper.to_array(ini)
                return arr
        for node in model.graph.node:
            if node.output[0] == constant_name:
                arr = onnx.numpy_helper.to_array(node.attribute[0].t)
                return arr

    def fix_node_weights(self, model_without_muls, weights_to_fix):
        modified_graph = model_without_muls
        for (node_name,constant_name) in weights_to_fix:
            numpy_weight = self.fetch_weights(model_without_muls, constant_name)
            x = torch.from_numpy(numpy_weight)
            x_diag = torch.diag(x)
            modified_graph = modify(
                onnx_graph=modified_graph,
                op_name=node_name,
                input_constants={constant_name: x_diag.numpy()}
            )
        return modified_graph

    def replace_mul_with_matmul(self, model):
        weights_to_fix = []
        # Iterate through all nodes in the model
        for node in model.graph.node:
            if node.op_type == "Mul":
                # Replace "Mul" node with "MatMul"
                node.op_type = "MatMul"
                for i in node.input:
                    if "Constant" in i or "trainable_layers.2.scale" in i:
                        weights_to_fix  = weights_to_fix + [(node.name, i)]
        # Save the modified model
        return model, weights_to_fix


    def conduct(self, report_dir: os.PathLike, storage_path: os.PathLike = None):
        model = onnx.load(self.path_flow)
        classifier = onnx.load(self.path_classifier)

        # save input models into the reports target folder.
        # Might be redundant in the sense that the input models are most likely already saved elsewhere.
        # This way we can make sure that
        curtime = str(datetime.now()).replace(" ", "")
        dir = f'{report_dir}/{self.name}/{curtime}/'
        os.makedirs(dir)
        onnx.save(model, f'{dir}/unmodified_model.onnx')
        onnx.save(classifier, f'{dir}/classifier.onnx')

        # First replace each node of type mul with matmul and save the names of its constant parameters.
        only_matmuls, weights_to_fix = self.replace_mul_with_matmul(model)
        # replace these constant parameters with their corresponding diagonals.
        fixed_weights_model = self.fix_node_weights(only_matmuls,weights_to_fix)
        onnx.checker.check_model(model=fixed_weights_model, full_check=True)

        errors = self.compare_models(self.path_flow, fixed_weights_model)
        if errors >0:
            print(Fore.RED + 'Model has errors! do not use it.')
        else:
            fixed_weights_model = self.convert_to_IR_8(fixed_weights_model)
            flow_output = fixed_weights_model.graph.output[0].name
            classifier_input = classifier.graph.input[0].name
            combined_model = onnx.compose.merge_models(
                fixed_weights_model, classifier,
                io_map=[(flow_output, classifier_input)]
            )
            onnx.checker.check_model(model=fixed_weights_model, full_check=True)
            onnx.save(combined_model, f'{dir}/merged_model.onnx')

        try:
            sys.path.append('/home/mustafa/repos/Marabou')
            from maraboupy import Marabou
        except ImportError:
            Marabou = None
        if Marabou:
            print(f'Marabou available! {Marabou}')
        else:
            print(Fore.RED + 'Marabou not found!')
