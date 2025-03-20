import onnx
import onnxruntime as ort
import torch


@staticmethod
def remove_reshape_node(model):
    # Ensures that the second input to the mul node is the constant node (and not the variable).
    # This constraint is imposed by the marabou library that implicitly
    # assumes the first input to be the variable and the second the constant factor.
    nodes_to_remove = []
    constant_outputs_to_remove = []
    for reshape_search_index in range(len(model.graph.node)):
        if model.graph.node[reshape_search_index].op_type == "Reshape":

            for recipient_search_index in range(len(model.graph.node)):
                for input_index_recipient in range(len(model.graph.node[recipient_search_index].input)):
                    if (model.graph.node[recipient_search_index].input[input_index_recipient] ==
                            model.graph.node[reshape_search_index].output[0]):

                        for input_index_reshape in range(len(model.graph.node[reshape_search_index].input)):
                            if "Constant" not in model.graph.node[reshape_search_index].input[input_index_reshape]:
                                model.graph.node[recipient_search_index].input[input_index_recipient] \
                                    = model.graph.node[reshape_search_index].input[input_index_reshape]
                                nodes_to_remove.append(model.graph.node[reshape_search_index])
                            else:
                                constant_outputs_to_remove.append(model.graph.node[reshape_search_index].input[input_index_reshape])
    for output_name in constant_outputs_to_remove:
        for constant_search_index in range(len(model.graph.node)):
            if model.graph.node[constant_search_index].output[0] == output_name:
                nodes_to_remove.append(model.graph.node[constant_search_index])

    for node in nodes_to_remove:
        model.graph.node.remove(node)
    return model

def compare_networks(PATH_APPROVED, PATH_TO_TEST, dimensions):
    model_approved = onnx.load(PATH_APPROVED)
    onnx.checker.check_model(model=model_approved, full_check=True)
    print("done checking consistency first model")
    model_modified = onnx.load(PATH_TO_TEST)
    onnx.checker.check_model(model=model_modified, full_check=True)
    print("done checking consistency second model")

    ort_sess_approved = ort.InferenceSession(PATH_APPROVED)
    ort_sess_modified = ort.InferenceSession(PATH_TO_TEST)
    diff_counter = 0
    random_test_counts = 1000
    for i in range(random_test_counts):
        x = torch.rand(dimensions)
        outputs_approved = ort_sess_approved.run(None, {'onnx::MatMul_0': x.numpy()})
        outputs_modified = ort_sess_modified.run(None, {'onnx::MatMul_0': x.numpy()})
        for i in range(len(outputs_approved[0])):
            diff = abs(outputs_approved[0][i] - outputs_modified[0][i])
            if diff > 0.001:
                diff_counter = diff_counter + 1
                print(diff)
    print(f'diff counter: {diff_counter}')
    if diff_counter == 0:
        print(f'all good, enjoy your new model.')
    else:
        print(f'Diff found. Check if either model transformation is broken, or, rounding errors may have occured.')


if __name__ == '__main__':
    dir_to_flow = '/home/mustafa/Documents/midas/flows_for_plausible_ctx/concrete_diabetes_cancer/1_cancer_flow/2024-11-1523:35:06.711882/'
    path_flow = f'{dir_to_flow}/forward.onnx'
    model = onnx.load(path_flow)
    dimensions = model.graph.input[0].type.tensor_type.shape.dim[0].dim_value
    model = remove_reshape_node(model)
    onnx.checker.check_model(model=model, full_check=True)
    target_flow_path = f'{dir_to_flow}/forward_processed.onnx'
    onnx.save(model, target_flow_path)
    print(f'Testing semantic equivalence of original and transformed model')
    compare_networks(path_flow, target_flow_path, dimensions)




