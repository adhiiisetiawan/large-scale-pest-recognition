from .mobilenetv2 import mobilenetV2
from torch import nn


class InsectPestClassifier(nn.Module):
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