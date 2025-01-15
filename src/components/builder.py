from __future__ import print_function
import os
import torch
from torchvision import models
import torch.nn as nn
import argparse
from components.se_sep_transformer import SeSepViT
from vit_pytorch.sep_vit import SepViT
from components.senet.se_resnet import se_resnet18, se_resnet20, se_resnet34, se_resnet50, se_resnet56, se_resnet101, se_resnet152
import timm


def build():
    # Parse command line arguments
    model_choice = ["alexnet", "resnet18", "resnet34", "resnet50", "seresnet18", "seresnet20", "seresnet34", "seresnet50", 
                    "seresnet56", "seresnet101", "seresnet152", "vgg13", "vgg16", "vgg19", "densenet121", "densenet161", "densenet169", 
                    "sepvit_lite", "sepvit_tiny", "sepvit_small", "sepvit_base", "sesepvit_lite_s", "sesepvit_lite", "sesepvit_tiny",
                    "sesepvit_small", "sesepvit_base", "vit", "vit_11", "vit_10", "vit_9", "vit_8", "vit_7", "vit_6", "vit_5", "vit_4", "fusion"]
# python .\slide_test.py --model-name resnet18 --dataset E:\patch\validation\ --save-path ..\evaluation_results\ --weights ..\finetuned_weights\resnet18.pth --confidence ..\confidence\ --batch-size 2048 --data-parallel
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, choices=model_choice, help="Model to train")
    parser.add_argument("--save-path", required=False, default="..\\evaluation_results", help="Path to save results like trained weights for train.py, confusion matrix for slide_test.py or feature map for cam.py")
    parser.add_argument("--dataset", required=False, help="Path of dataset")
    parser.add_argument("--weights", required=False, default= "..\\finetuned_weights\\", help="Path of finetuned weights, used by slide_test.py.")
    parser.add_argument("--pretrained-weights", required=False, default="..\\pretrained_weights\\", help="Path of pretrained weights, used by train.py")
    parser.add_argument("--confidence", required=False, default="..\\confidence\\", help="Path of pre-computed preds and confidence, used by slide_test.py")
    parser.add_argument("--batch-size", required=False, default=256, help="Batch size")
    parser.add_argument("--breakpoint", required=False, help="Continuing training from last breakpoint")
    parser.add_argument("--epoch", required=False, default=20, help="Epoch")
    parser.add_argument("--lr", required=False, default=0.001, help="Learning rate")
    parser.add_argument("--data-parallel", action= "store_true", help="Use multiple GPUs")
    parser.add_argument("--device", required=False, default="cuda", help="Choose which GPU to use")
    parser.add_argument("--image-path", type=str, default="", help="Input image path, only valid for cam.py")
    args = parser.parse_args()
    args.weights = os.path.join(args.weights, args.model_name + ".pth")
    args.pretrained_weights = os.path.join(args.pretrained_weights, args.model_name + ".pth")

    model = None

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if args.model_name == "alexnet":
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    elif args.model_name == "vgg16":
        model = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
    elif args.model_name == "vgg13":
        model = models.vgg13_bn(weights=models.VGG13_BN_Weights.DEFAULT)
    elif args.model_name == "vgg19":
        model = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT)
    elif args.model_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    elif args.model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif args.model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif args.model_name == "seresnet18":
        model = se_resnet18(num_classes=1000)
    elif args.model_name == "seresnet20":
        model = se_resnet20(num_classes=1000)
    elif args.model_name == "seresnet34":
        model = se_resnet34(num_classes=1000)
    elif args.model_name == "seresnet50":
        model = se_resnet50(num_classes=1000)
    elif args.model_name == "seresnet56":
        model = se_resnet56(num_classes=1000)
    elif args.model_name == "seresnet101":
        model = se_resnet101(num_classes=1000)
    elif args.model_name == "seresnet152":
        model = se_resnet152(num_classes=1000)
    elif args.model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    elif args.model_name == "densenet161":
        model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
    elif args.model_name == "densenet169":
        model = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
    elif args.model_name == "vit":
        model = timm.create_model('vit_base_patch8_224', pretrained=True)
        model.head = nn.Linear(in_features=model.head.in_features, out_features=3)
    elif args.model_name[:3] == "vit":
        model = timm.create_model('vit_base_patch8_224', pretrained=True)
        model.blocks = model.blocks[:int(args.model_name[4:])]
        model.head = nn.Linear(in_features=model.head.in_features, out_features=3)
    elif args.model_name == "sesepvit_lite_s":
        model = SeSepViT(
            num_classes = 1000,
            dim = 12,               # dimensions of first stage, which doubles every stage
            dim_head = 12,          # attention head dimension
            heads = (1, 2, 2, 4),   # number of heads per stage
            depth = (1, 2, 4, 2),   # number of transformer blocks per stage
            window_size = 7,        # window size of DSS Attention block
            dropout = 0.1           # dropout
        )
    elif args.model_name == "sesepvit_lite":
        model = SeSepViT(                                                               
        num_classes = 1000,
        dim = 32,               
        dim_head = 32,         
        heads = (1, 2, 4, 8),   
        depth = (1, 2, 6, 2),  
        window_size = 7,      
        dropout = 0.1        
    )
    elif args.model_name == "sesepvit_tiny":
        model = SeSepViT(                                                               
        num_classes = 1000,
        dim = 32,             
        dim_head = 32,        
        heads = (3, 6, 12, 24),
        depth = (1, 2, 6, 2), 
        window_size = 7,     
        dropout = 0.1      
    )
    elif args.model_name == "sesepvit_small":
        model = SeSepViT(                                                               
        num_classes = 1000,
        dim = 32,             
        dim_head = 32,       
        heads = (3, 6, 12, 24),
        depth = (1, 2, 14, 2), 
        window_size = 7,      
        dropout = 0.1     
    )
    elif args.model_name == "sesepvit_base":
        model = SeSepViT(                                                               
        num_classes = 1000,
        dim = 32,            
        dim_head = 32,   
        heads = (4, 8, 16, 32),
        depth = (1, 2, 14, 2),
        window_size = 7,      
        dropout = 0.1     
    )
    elif args.model_name == "sepvit_lite":
        model = SepViT(                                                               
        num_classes = 1000,
        dim = 32,           
        dim_head = 32,          
        heads = (1, 2, 4, 8),  
        depth = (1, 2, 6, 2),  
        window_size = 7,       
        dropout = 0.1          
    )
    elif args.model_name == "sepvit_tiny":
        model = SepViT(   
        num_classes = 1000,
        dim = 32,               
        dim_head = 32,        
        heads = (3, 6, 12, 24), 
        depth = (1, 2, 6, 2), 
        window_size = 7,      
        dropout = 0.1         
    )
    elif args.model_name == "sepvit_small":
        model = SepViT(                                                               
        num_classes = 1000,
        dim = 32,            
        dim_head = 32,      
        heads = (3, 6, 12, 24),  
        depth = (1, 2, 14, 2), 
        window_size = 7,      
        dropout = 0.1        
    )
    elif args.model_name == "sepvit_base":
        model = SepViT(                                                               
        num_classes = 1000,
        dim = 32,            
        dim_head = 32,       
        heads = (4, 8, 16, 32),  
        depth = (1, 2, 14, 2),  
        window_size = 7,      
        dropout = 0.1       
    )

    if args.model_name[:3] == "vgg" or args.model_name == "alexnet":
        model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features, out_features=3)
    elif args.model_name[:8] == "densenet":
        model.classifier = nn.Linear(in_features=model.classifier.in_features, out_features=3)
    elif args.model_name[:6] == "resnet" or args.model_name[:8] == "seresnet":
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=3)
    elif args.model_name[:6] == "sepvit" or args.model_name[:8] == "sesepvit":
        model.mlp_head[-1] = nn.Linear(model.mlp_head[-1].in_features, 3)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    return model, args

if __name__ == "__main__":
    build()