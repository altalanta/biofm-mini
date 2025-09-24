import torch

from biofm.models.clip import BioFMClipModel
from biofm.models.encoders import ImageEncoder, RNAMlpEncoder


def test_clip_step_backward() -> None:
    model = BioFMClipModel(
        image_encoder=ImageEncoder(embedding_dim=32, pretrained=False),
        rna_encoder=RNAMlpEncoder(input_dim=16, embedding_dim=32),
        projector_dim=32,
        temperature=0.1,
    )
    pixel_values = torch.randn(4, 3, 64, 64)
    expression = torch.randn(4, 16)
    outputs = model(pixel_values=pixel_values, expression=expression)
    loss = outputs["loss"]
    loss.backward()
    assert loss.item() > 0
