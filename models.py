import torch
import torch.nn as nn
import torchvision.models as models
from config import Config


class SSDModel(nn.Module):
    def __init__(self, num_classes):
        super(SSDModel, self).__init__()

        # Use ResNet50 as backbone
        backbone = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(backbone.children())[:-2])

        # SSD specific layers
        self.conv1 = nn.Conv2d(2048, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        # Classification head
        self.clf = nn.Sequential(
            nn.Conv2d(256, num_classes * 6, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # Bounding box regression head
        self.reg = nn.Sequential(
            nn.Conv2d(256, 4 * 6, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # Feature extraction
        x = self.features(x)

        # SSD specific features
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Get classifications and bounding boxes
        classifications = self.clf(x)
        bboxes = self.reg(x)

        return classifications, bboxes


class YOLOModel(nn.Module):
    def __init__(self, num_classes):
        super(YOLOModel, self).__init__()

        # Darknet-like backbone
        self.features = nn.Sequential(
            self._conv_block(3, 32, 3),
            self._conv_block(32, 64, 3),
            self._conv_block(64, 128, 3),
            self._conv_block(128, 256, 3),
            self._conv_block(256, 512, 3),
            self._conv_block(512, 1024, 3),
        )

        # YOLO detection heads
        self.det1 = self._detection_block(1024, num_classes)
        self.det2 = self._detection_block(512, num_classes)
        self.det3 = self._detection_block(256, num_classes)

    def _conv_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2)
        )

    def _detection_block(self, in_channels, num_classes):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.Conv2d(in_channels // 2, (num_classes + 5) * 3, 3, padding=1)
        )

    def forward(self, x):
        features = []
        for layer in self.features:
            x = layer(x)
            features.append(x)

        # Detection at different scales
        det1 = self.det1(features[-1])
        det2 = self.det2(features[-2])
        det3 = self.det3(features[-3])

        return [det1, det2, det3]


def get_model(model_type, num_classes=Config.NUM_CLASSES):
    if model_type.lower() == 'ssd':
        return SSDModel(num_classes)
    elif model_type.lower() == 'yolo':
        return YOLOModel(num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")