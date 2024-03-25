import onnx
import os
import sys
import onnxruntime as ort
import numpy

from datetime import datetime
from src.explib.base import Experiment
from colorama import Fore
from matplotlib import pyplot as plt
from scipy.stats import norm
import torch
import math


class OnnxConverter(Experiment):

    def __init__(self, path_flow: str, path_classifier: str, test_with_flow: bool,  *args, **kwargs,) -> None:
        """Initialize verification experiment.
        Args:
            path_flow (string): The path to the flow model used for the verification experiment in ONNX format.
            path_classifier (string): Full path to the classifier in ONNX format.
        """
        super().__init__(*args, **kwargs)
        self.path_flow = path_flow
        self.path_classifier = path_classifier
        self.test_with_flow = test_with_flow


    def quantile_log_normal(self, p, mu=1, sigma=0.5):
        return math.exp(mu + sigma * norm.ppf(p))


    def verify_classifier_only(self, classifier_path, maraboupy, directory):
        options = maraboupy.Marabou.createOptions(verbosity=1)
        network = maraboupy.Marabou.read_onnx(classifier_path)
        input_vars = network.inputVars[0]
        # Add 100 additional variables that will encode the abs of the input vars.
        for i in range(len(network.inputVars[0])):
            # sets the value of the new variables as the abs value of the input
            network.setLowerBound(i, 0)
            network.setUpperBound(i, 255)
        # Adds inequality: (SUM_i (redundant_vars[i] * ones[i]))  <= threshold_input

        var = maraboupy.MarabouPythonic.Var
        output_vars = network.outputVars[0][0]
        target_class = 0
        # Add inequalities that ensure that the input is classified as 0
        for i in range(len(output_vars)):
            if not i == target_class:
                network.addConstraint(var(output_vars[target_class]) - var(output_vars[i]) >= 0.01)

        # Now add the confidence term that ensures that the input is classified with high confidence.
        # Since we look for counter examples, check for violations. I.e. instances where the
        # confidence is lower than indicated by the threshold.
        coefficients_classifier = [1 if i == target_class else -1/10 for i in range(len(output_vars))]
        print(coefficients_classifier)
        confidence_threshold = 5.0
        # less or equal inequality. (SUM_{0<=i<=9}coefficients_classifier[i]*output_vars) <= confidence_threshold
        network.addInequality(output_vars, coefficients_classifier, confidence_threshold)
        vals = network.solve(filename =f'{directory}/marabou-output.txt', options=options)
        assignments = [vals[1][i] for i in range(100)]
        ort_sess_classifier = ort.InferenceSession(classifier_path)
        outputs_combined_model = ort_sess_classifier.run(
            None,
            {'onnx::MatMul_0': numpy.asarray(assignments).astype(numpy.float32)})  #before x.1
        print(f' outputs of the classifier using onnxruntime: {outputs_combined_model}')
        logits = numpy.asarray(outputs_combined_model[0]).astype(numpy.float32)
        print(f'confidence: {(len(logits) * logits[target_class] - (numpy.sum(logits) - logits[target_class])) / len(logits)}')
        return numpy.asarray(assignments).astype(numpy.float32)



    def verify_merged_model(self, combined_model_path, maraboupy, directory):
        options = maraboupy.Marabou.createOptions(verbosity=1)
        network = maraboupy.Marabou.read_onnx(combined_model_path)
        input_vars = network.inputVars[0]
        threshold_input = self.quantile_log_normal(p=0.01)# ~ central p fraction of the radial distribution
        print(f'threshold_input: {threshold_input}')
        num_vars = network.numVars
        redundant_var_count = 100  # number of input vars to the network. in our case 10*10.
        redundant_vars = [i for i in range(num_vars, num_vars + redundant_var_count)]
        ones = [1.0 for i in range(len(redundant_vars))]
        # Add 100 additional variables that will encode the abs of the input vars.
        network.numVars = num_vars + redundant_var_count
        for i in range(redundant_var_count):
            # sets the value of the new variables as the abs value of the input
            network.addAbsConstraint(i, redundant_vars[i])
        # Adds inequality: (SUM_i (redundant_vars[i] * ones[i]))  <= threshold_input
        network.addInequality(redundant_vars, ones, threshold_input)

        var = maraboupy.MarabouPythonic.Var
        output_vars = network.outputVars[0][0]
        target_class = 0
        # Add inequalities that ensure that the input is classified as 0
        for i in range(len(output_vars)):
            if not i == target_class:
                network.addConstraint(var(output_vars[target_class]) - var(output_vars[i]) >= 0.001)

        # Now add the confidence term that ensures that the input is classified with high confidence.
        # Since we look for counter examples, check for violations. I.e. instances where the
        # confidence is lower than indicated by the threshold.
        coefficients_classifier = [1 if i == target_class else -1/10 for i in range(len(output_vars))]
        print(coefficients_classifier)
        confidence_threshold = 5.0
        # less or equal inequality. (SUM_{0<=i<=9}coefficients_classifier[i]*output_vars) <= confidence_threshold
        network.addInequality(output_vars, coefficients_classifier, confidence_threshold)

        vals = network.solve(filename =f'{directory}/marabou-output.txt', options=options)

        assignments = [vals[1][i] for i in range(100)]
        print(f'sum of assignments  {sum([abs(ele) for ele in assignments])}')
        ort_sess_classifier = ort.InferenceSession(combined_model_path)
        outputs_combined_model = ort_sess_classifier.run(
            None,
            {'onnx::MatMul_0': numpy.asarray(assignments).astype(numpy.float32)})  #before x.1
        print(f' outputs of the combined model using onnxruntime: {outputs_combined_model}')
        logits = numpy.asarray(outputs_combined_model[0]).astype(numpy.float32)
        print(f'max at position {numpy.argmax(logits)} and target is {target_class}')
        print(f'confidence: {(len(logits) * logits[target_class] - (numpy.sum(logits) - logits[target_class])) / len(logits)}')

        return numpy.asarray(assignments).astype(numpy.float32)

    @staticmethod
    def swap_mul_inputs(model):
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
        directory = f'  {report_dir}/{self.name}/{current_time}'
        os.makedirs(directory)
        return directory

    @staticmethod
    def save_model(model, directory, model_name):
        saved_path = f'{directory}/{model_name}'
        onnx.save(model, f'{directory}/{model_name}')
        return saved_path

    @staticmethod
    def merge_models(modified_model, classifier):
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
        unmodified_model_path = self.save_model(model, directory, "unmodified_model.onnx")
        classifier_path = self.save_model(classifier, directory, "classifier.onnx")
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
                if self.test_with_flow:
                    counter_example = self.verify_merged_model(combined_model_path, maraboupy, directory)
                    ort_sess_classifier = ort.InferenceSession(unmodified_model_path)
                    outputs_flow_image = ort_sess_classifier.run(
                        None,
                        {'onnx::MatMul_0': counter_example})  # before x.1
                    print(f'outputs_flow_image {outputs_flow_image}')
                    sample = outputs_flow_image
                    sample = numpy.uint8(numpy.clip(sample, 0, 1) * 255)
                    plt.imshow(torch.tensor(sample).view(10, 10), cmap='gray')
                    plt.savefig(f'{directory}/counterexample.png')
                    numpy.save(file=f'{directory}/counter_example.npy', arr=counter_example)
                    numpy.savetxt(f'{directory}/counter_example.txt', counter_example)

                    ort_sess_classifier = ort.InferenceSession(classifier_path)
                    outputs_classifier = ort_sess_classifier.run(
                        None,
                        {'onnx::MatMul_0': numpy.array(outputs_flow_image[0])})
                    print(f'outputs_classifier {outputs_classifier}')
                    numpy.save(file=f'{directory}/classifications.npy', arr=outputs_classifier)
                    numpy.savetxt(f'{directory}/classifications.txt', outputs_classifier)
                else:
                    counter_example = self.verify_classifier_only(classifier_path, maraboupy, directory)
                    sample = counter_example
                    sample = numpy.uint8(numpy.clip(sample, 0, 1) * 255)
                    plt.imshow(torch.tensor(sample).view(10, 10), cmap='gray')
                    plt.savefig(f'{directory}/counterexample.png')
                    numpy.save(file=f'{directory}/counter_example.npy', arr=counter_example)
                    numpy.savetxt(f'{directory}/counter_example.txt', counter_example)

                    ort_sess_classifier = ort.InferenceSession(classifier_path)
                    outputs_classifier = ort_sess_classifier.run(
                        None,
                        {'onnx::MatMul_0': numpy.array(counter_example)})
                    print(f'outputs_classifier {outputs_classifier}')
                    numpy.save(file=f'{directory}/classifications.npy', arr=outputs_classifier)
                    numpy.savetxt(f'{directory}/classifications.txt', outputs_classifier)

        else:
            print(Fore.RED + 'Marabou not found!')
