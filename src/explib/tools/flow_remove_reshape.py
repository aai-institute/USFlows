import onnx


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

if __name__ == '__main__':
    path_flow = "models/model_0_lognormal_decay_forward.onnx"
    model = onnx.load(path_flow)
    model = remove_reshape_node(model)
    onnx.checker.check_model(model=model, full_check=True)
    onnx.save(model, "models/without_reshape.onnx")
