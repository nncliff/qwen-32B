import onnx
import numpy as np
from onnx import helper, checker, numpy_helper
import copy

# create a simple ONNX model
def create_simple_model():
    # Create a simple ONNX model that adds two inputs
    input_tensor = helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [None, 4])
    output_tensor = helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [None, 4])
    const_tensor = numpy_helper.from_array(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), name='const_tensor')

    # Create an Identity node (operation)
    identity_node = helper.make_node(
        'Identity',
        inputs=['input'],
        outputs=['identity_output'],
        name='IdentityNode'
    )

    # Create an Add node
    add_node = helper.make_node(
        'Add',
        inputs=['identity_output', 'const_tensor'],
        outputs=['add_output'],
        name='AddNode'
    )

    # Create a Relu node
    relu_node = helper.make_node(
        'Relu',
        inputs=['add_output'],
        outputs=['output'],
        name='ReluNode'
    )

    # Build the graph
    graph = helper.make_graph(
        nodes=[identity_node, add_node, relu_node],
        name='SimpleAddGraph',
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[const_tensor]
    )

    model = helper.make_model(graph)
    checker.check_model(model)
    return model

def remove_identity_nodes(model):
    new_nodes = []
    for node in model.graph.node:
        if node.op_type != 'Identity':
            new_nodes.append(node)
        else:
            # Bypass the Identity node by connecting its input directly to its output
            input_name = node.input[0]
            output_name = node.output[0]
            for n in model.graph.node:
                for i, name in enumerate(n.input):
                    if name == output_name:
                        n.input[i] = input_name
    
    model.graph.ClearField('node')
    model.graph.node.extend(new_nodes)
    return model

def fuse_add_relu(model):
    new_nodes = []
    skip_next = False
    for i in range(len(model.graph.node)):
        if skip_next:
            skip_next = False
            continue
        node = model.graph.node[i]

        if node.op_type == 'Add' and i + 1 < len(model.graph.node):
            next_node = model.graph.node[i + 1]
            # Do we still need to check input/output names to be sure they are connected?
            if next_node.op_type == 'Relu' and next_node.input[0] == node.output[0]:
                # Create a fused AddRelu node
                fused_node = helper.make_node(
                    'FusedAddRelu',
                    inputs=node.input,
                    outputs=next_node.output,
                    name='Fused_' + node.name + '_' + next_node.name
                )
                new_nodes.append(fused_node)
                skip_next = True
            else:
                new_nodes.append(node)
        else:
            new_nodes.append(node)
    
    model.graph.ClearField('node')
    model.graph.node.extend(new_nodes)
    return model

def constant_folding_pass(model):
    # A very simplified constant folding that only handles Add nodes with constant inputs
    new_model = copy.deepcopy(model)
    initializer_map = {init.name: numpy_helper.to_array(init) for init in new_model.graph.initializer}

    new_nodes = []
    for node in new_model.graph.node:
        if node.op_type == 'Add':
            input_arrays = []
            all_constants = True
            for input_name in node.input:
                if input_name in initializer_map:
                    input_arrays.append(initializer_map[input_name])
                else:
                    found = False
                    for prev_node in new_model.graph.node:
                        if prev_node.output[0] == input_name and prev_node.op_type == 'Identity':
                            if prev_node.input[0] in initializer_map:
                                input_arrays.append(initializer_map[prev_node.input[0]])
                                found = True
                                break
                    if not found:
                        all_constants = False
                        break

            if all_constants and len(input_arrays) > 0:
                result = sum(input_arrays)
                new_init = numpy_helper.from_array(result.astype(np.float32), name=node.output[0])
                initializer_map[node.output[0]] = result
                new_model.graph.initializer.append(new_init)
                continue  # Skip adding this node, as it's replaced by a constant

        new_nodes.append(node)
    new_model.graph.ClearField('node')
    new_model.graph.node.extend(new_nodes)
    return new_model

def run_pass_chain(model, passes):
    optimized_model = model
    for p in passes:
        optimized_model = p(optimized_model)
    return optimized_model

def main():
    model = create_simple_model()
    print("Original Model:")
    for node in model.graph.node:
        print(f" - {node.name}: {node.op_type}")
    
    passes = [remove_identity_nodes, fuse_add_relu, constant_folding_pass]
    optimized_model = run_pass_chain(model, passes)

    print("\nOptimized Model:")
    for node in optimized_model.graph.node:
        print(f" - {node.name}: {node.op_type}")
    
    optimized_model_path = "optimized_model.onnx"
    onnx.save(optimized_model, optimized_model_path)
    print(f"\nOptimized model saved to {optimized_model_path}")

if __name__ == "__main__":
    main()