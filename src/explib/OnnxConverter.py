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

    def __init__(self, path_flow: str, path_classifier: str, verify_within_dist: bool,  *args, **kwargs,) -> None:
        """Initialize verification experiment.
        Args:
            path_flow (string): The path to the flow model used for the verification experiment in ONNX format.
            path_classifier (string): Full path to the classifier in ONNX format.
        """
        super().__init__(*args, **kwargs)
        self.path_flow = path_flow
        self.path_classifier = path_classifier
        self.verify_within_dist = verify_within_dist


    def quantile_log_normal(self, p, mu=1, sigma=0.5):
        return math.exp(mu + sigma * norm.ppf(p))

    def add_post_condition(self, network, target_class, maraboupy, confidence_threshold):
        var = maraboupy.MarabouPythonic.Var
        output_vars = network.outputVars[0][0]
        # Add inequalities that ensure that the input is classified as target_class
        for i in range(len(output_vars)):
            if not i == target_class:
                network.addConstraint(var(output_vars[target_class]) - var(output_vars[i]) >= 0.001)

        # Now add the confidence term that ensures that the input is classified with high confidence.
        # Since we look for counter examples, check for violations. I.e. instances where the
        # confidence is lower than indicated by the threshold.
        coefficients_classifier = [1 if i == target_class else -1/10 for i in range(len(output_vars))]
        # less or equal inequality. (SUM_{0<=i<=9}coefficients_classifier[i]*output_vars) <= confidence_threshold
        network.addInequality(output_vars, coefficients_classifier, confidence_threshold)

    def verify_classifier_only(self, classifier_path, maraboupy, directory, target_class, confidence_threshold):
        network = maraboupy.Marabou.read_onnx(classifier_path)
        for i in range(len(network.inputVars[0])):
            network.setLowerBound(i, 0)
            network.setUpperBound(i, 255)
        self.add_post_condition(network, target_class, maraboupy, confidence_threshold)
        vals = network.solve(filename =f'{directory}/marabou-output.txt',
                             options=maraboupy.Marabou.createOptions(verbosity=1, timeoutInSeconds=60))
        if vals[0] == 'TIMEOUT':
            print(f'did not solve experiment {confidence_threshold} without flow')
            return numpy.asarray([]), True

        assignments = [vals[1][i] for i in range(100)]
        return numpy.asarray(assignments).astype(numpy.float32), False


    def verify_merged_model(self, combined_model_path, maraboupy, directory, target_class , confidence_threshold):
        network = maraboupy.Marabou.read_onnx(combined_model_path)
        threshold_input = self.quantile_log_normal(p=0.01)# ~ central p fraction of the radial distribution
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

        self.add_post_condition(network, target_class, maraboupy, confidence_threshold)

        vals = network.solve(filename =f'{directory}/marabou-output.txt',
                             options=maraboupy.Marabou.createOptions(verbosity=1, timeoutInSeconds=60))
        if vals[0] == 'TIMEOUT':
            print(f'did not solve experiment {confidence_threshold} with flow')
            return numpy.asarray([]), True

        assignments = [vals[1][i] for i in range(100)]
        sum_of_assignments = sum([abs(ele) for ele in assignments])
        if sum_of_assignments > threshold_input + 0.001:
            print(Fore.RED + f'ERROR: sum of abs assignments exceeds threshold')
            sys.exit(0)
        return numpy.asarray(assignments).astype(numpy.float32), False


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
        directory_with_flow = f'{directory}/with_flow'
        os.makedirs(directory_with_flow)
        directory_without_flow = f'{directory}/without_flow'
        os.makedirs(directory_without_flow)
        return directory, directory_with_flow, directory_without_flow

    def create_experiment_subdir(self, directory, folder_name):
        subdirectory = f'{directory}/{folder_name}'
        os.makedirs(subdirectory)
        return subdirectory

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



    def fill_report(self, image, counter_example, directory, classifier_path, target_class):
        sample = image
        sample = numpy.uint8(numpy.clip(sample, 0, 1) * 255)
        plt.imshow(torch.tensor(sample).view(10, 10), cmap='gray')
        plt.savefig(f'{directory}/counterexample.png')
        numpy.save(file=f'{directory}/counter_example.npy', arr=counter_example)
        numpy.savetxt(f'{directory}/counter_example.txt', counter_example)
        ort_sess_classifier = ort.InferenceSession(classifier_path)
        outputs_classifier = ort_sess_classifier.run(
            None,
            {'onnx::MatMul_0': numpy.array(image)})
        logits = numpy.asarray(outputs_classifier[0]).astype(numpy.float32)
        confidence = (len(logits) * logits[target_class] - (numpy.sum(logits) - logits[target_class])) / len(logits)
        classified_as = numpy.argmax(logits)
        result_property_check = numpy.asarray([confidence, classified_as, target_class])
        numpy.save(file=f'{directory}/classifications.npy', arr=outputs_classifier)
        numpy.savetxt(f'{directory}/classifications.txt', outputs_classifier)
        numpy.savetxt(f'{directory}/classification_confidence.txt', result_property_check)


    def run_within_distribution_verification(self, directory_with_flow, combined_model_path, maraboupy, target_class,
                                             unmodified_model_path, classifier_path, directory_without_flow):

        for confidence_threshold in range(1, 21, 1):
            print(f'experiment no {confidence_threshold} of 20')
            experiment_directory_with_flow = self.create_experiment_subdir(directory_with_flow, confidence_threshold)
            counter_example, is_error = self.verify_merged_model(combined_model_path, maraboupy,
                                                                 experiment_directory_with_flow, target_class,
                                                                 confidence_threshold)
            if is_error:
                continue
            outputs_flow_image = ort.InferenceSession(unmodified_model_path).run(
                None,
                {'onnx::MatMul_0': counter_example})
            self.fill_report(outputs_flow_image[0], counter_example, experiment_directory_with_flow, classifier_path,
                             target_class)

            experiment_directory_without_flow = self.create_experiment_subdir(directory_without_flow,
                                                                              confidence_threshold)
            counter_example, is_error = self.verify_classifier_only(classifier_path, maraboupy,
                                                                    experiment_directory_without_flow, target_class,
                                                                    confidence_threshold)
            if is_error:
                continue
            self.fill_report(counter_example, counter_example, experiment_directory_without_flow, classifier_path,
                             target_class)

    def conduct(self, report_dir: os.PathLike, storage_path: os.PathLike = None):
        model = onnx.load(self.path_flow)
        classifier = onnx.load(self.path_classifier)
        directory, directory_with_flow, directory_without_flow = self.fetch_directory(report_dir)
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
            target_class = 9
            if combined_model:
                if self.verify_within_dist:
                    self.run_within_distribution_verification(directory_with_flow, combined_model_path, maraboupy,
                                                              target_class, unmodified_model_path, classifier_path,
                                                              directory_without_flow)

            else:
                print(Fore.RED + 'Error combining models has failed')
        else:
            print(Fore.RED + 'Marabou not found!')
