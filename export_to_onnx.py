import torch
from model import MiniESPCN

model = MiniESPCN(upscale_factor=2)
model.load_state_dict(torch.load("miniespcn.pth"))
model.eval()

dummy_input = torch.randn(1, 3, 180, 320)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)