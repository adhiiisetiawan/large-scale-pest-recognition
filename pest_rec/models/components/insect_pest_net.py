from .mobilenetv2 import mobilenetV2
from torch import nn


class InsectPestClassifier(nn.Module):
    """
    A classifier model based on MobileNetV2 architecture for insect pest recognition.

    The InsectPestClassifier is a neural network model that utilizes MobileNetV2 as its base
    feature extractor and adds additional fully connected layers for classification. The model
    can be optionally frozen to prevent updating the weights of the base feature extractor during
    training.

    Args:
        input_size (int, optional): The number of input features to the classifier. Default is 1280.
        linear1_size (int, optional): The number of units in the first fully connected layer. Default is 1024.
        linear2_size (int, optional): The number of units in the second fully connected layer. Default is 512.
        linear3_size (int, optional): The number of units in the third fully connected layer. Default is 256.
        output_size (int, optional): The number of output classes. Default is 102.
        dropout_size (float, optional): The dropout probability for the fully connected layers. Default is 0.2.
        freeze (bool, optional): Flag indicating whether to freeze the weights of the base feature extractor.
            If True, the weights are frozen; if False, the weights are trainable. Default is True.

    Attributes:
        mobilenet (nn.Module): The MobileNetV2 base feature extractor.
    
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Performs a forward pass through the network.
        freeze():
            Freezes the weights of the base feature extractor.
        unfreeze():
            Unfreezes the weights of the base feature extractor.
    """
    
    def __init__(
            self,
            input_size: int = 1280,
            linear1_size: int = 1024,
            linear2_size: int = 512,
            linear3_size: int = 256,
            output_size: int = 102,
            dropout_size: float = 0.2,
            freeze: bool = True

        ):
        super().__init__()

        self.mobilenet = mobilenetV2(pretrained=True)
        
        if freeze:
            self.freeze()
        else:
            self.unfreeze()
        
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(input_size, linear1_size),
            nn.ReLU(),
            nn.Dropout(dropout_size),

            nn.Linear(linear1_size, linear2_size),
            nn.ReLU(),
            nn.Dropout(dropout_size),

            nn.Linear(linear2_size, linear3_size),
            nn.ReLU(),
            nn.Dropout(dropout_size),

            nn.Linear(linear3_size, output_size)
        )
        
    def forward(self, x):
        return self.mobilenet(x)
    
    def freeze(self):
        for param in self.mobilenet.parameters():
            param.requires_grad = False
        
    def unfreeze(self):
        for param in self.mobilenet.parameters():
            param.requires_grad = True
