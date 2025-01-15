from components.full_attention import full_attention
import torch
import torch.nn as nn


class fusion(nn.Module):
    # dim等于model1和model2输出特征图的通道数之和
    def __init__(self, model1, model2, dim, num_classes=1000):
        super(fusion, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.dim = dim
        self.fusion_block = full_attention(dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        x = torch.concat([x1, x2], dim=1)
        x = self.fusion_block(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
        

    
if __name__ == "__main__":
    model1 = nn.Identity()
    model2 = nn.Identity()
    x = torch.randn(1, 64, 224, 224)
    fusion_model = fusion(model1, model2, 128)
    output = fusion_model(x)
    print(output.shape)