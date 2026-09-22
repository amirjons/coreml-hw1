import torch, torch.nn as nn, torchvision
import coremltools as ct

weights = torchvision.models.MobileNet_V2_Weights.DEFAULT
base_model = torch.hub.load("pytorch/vision", "mobilenet_v2", weights=weights).eval()

# Оборачиваем в Softmax, чтобы выход был настоящими вероятностями
model = nn.Sequential(base_model, nn.Softmax(dim=1)).eval()

example = torch.rand(1, 3, 224, 224)
traced = torch.jit.trace(model, example)

mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(
        name="image",
        shape=example.shape,
        scale=1 / (255 * 0.226),
        bias=[-0.485 / 0.226, -0.456 / 0.226, -0.406 / 0.226],
    )],
    classifier_config=ct.ClassifierConfig(weights.meta["categories"]),
    convert_to="mlprogram",
)
mlmodel.save("MobileNetV2.mlpackage")
print("готово")
