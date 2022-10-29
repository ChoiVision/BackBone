import torch
import torch.nn as nn
import torch.onnx

def export_onnx(model, dummy, output_dir):
    '''
    model : model
    dummy : dummy data ex) torch.rand((1,3,224,224))
    output_dir : ex) output.onnx
    '''
    torch.onnx.export(model, dummy, output_dir)