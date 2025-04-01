import torch.nn as nn
import torch

from pathlib import Path

class SimpleConvModel(nn.Module):

    def __init__(self):
        super(SimpleConvModel, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


def prepare_model(output_dir):
    torch.manual_seed(42)
    model = SimpleConvModel()

    dummy_input = torch.randn(1, 3, 4, 4)

    onnx_model_path = Path(output_dir, 'simple_conv_model.onnx').as_posix()
    quant_onnx_model_path = Path(output_dir, 'simple_conv_model_quantized.onnx').as_posix()

    torch.onnx.export(model,
                      dummy_input,
                      onnx_model_path,
                      input_names=['input'],
                      output_names=['output'],
                      opset_version=17)

    print(f'Model has been saved to {onnx_model_path}')
    return onnx_model_path, quant_onnx_model_path

class TransformerBlock(nn.Module):

    def __init__(self, emb_size, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(nn.Linear(emb_size, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, emb_size),
                                 nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbedding(nn.Module):

    def __init__(self, in_channels, patch_size, emb_size):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class ViT(nn.Module):

    def __init__(self, img_size=4, patch_size=2, in_channels=3, emb_size=64, num_heads=4, mlp_dim=128, num_classes=10):
        super(ViT, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.zeros(1, (img_size // patch_size)**2 + 1, emb_size))
        self.transformer = TransformerBlock(emb_size, num_heads, mlp_dim)
        self.mlp_head = nn.Sequential(nn.LayerNorm(emb_size), nn.Linear(emb_size, num_classes))

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = self.mlp_head(x[:, 0])
        return x


def prepare_model_vit(output_dir):
    torch.manual_seed(42)
    model = ViT(img_size=4, patch_size=2, in_channels=3, emb_size=64, num_heads=4, mlp_dim=128, num_classes=10)

    dummy_input = torch.randn(1, 3, 4, 4)

    onnx_model_path = Path(output_dir, 'vit_model.onnx').as_posix()
    onnx_quantized_model_path = Path(output_dir, 'vit_quantized.onnx').as_posix()
    torch.onnx.export(model,
                      dummy_input,
                      onnx_model_path,
                      input_names=['input'],
                      output_names=['output'],
                      opset_version=17)

    print(f'Model has been saved to {onnx_model_path}')
    return onnx_model_path, onnx_quantized_model_path
