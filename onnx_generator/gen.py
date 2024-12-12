import torch
import torch.nn as nn

class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.linear1 = nn.Linear(3, 4)
        self.linear2 = nn.Linear(4, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

# Create model
model = SimpleNetwork()

# Create dummy input (matching your C++ input size)
dummy_input = torch.randn(2, 3)  # batch_size=2, input_features=3

# Export the model
torch.onnx.export(model,                     # model being run
                 dummy_input,                # model input (or a tuple for multiple inputs)
                 "simple_linear_model.onnx", # where to save the model
                 export_params=True,         # store the trained parameter weights inside the model file
                 opset_version=12,           # the ONNX version to export the model to
                 do_constant_folding=True,   # whether to execute constant folding for optimization
                 input_names=['input'],      # the model's input names
                 output_names=['output'],    # the model's output names
                 dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                             'output': {0 : 'batch_size'}})

print("Model has been exported to simple_linear_model.onnx")