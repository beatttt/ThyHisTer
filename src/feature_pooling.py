import torchvision.models as models
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from components.dataset import slide_eval_dataset
from torchvision import transforms
from components.senet.se_resnet import se_resnet50
import numpy as np
import timm
from components.se_sep_transformer import SeSepViT
from vit_pytorch.sep_vit import SepViT
import os
import argparse


if __name__ == "__main__":
    # Parse command line arguments
    model_choice = ["resnet18", "resnet34", "vgg13", "vgg16", "vgg19", "sesepvit", "vit"]
    feature_length = {"vgg16": 25088, "resnet34": 512}

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, choices=model_choice, help="Model to train")
    parser.add_argument("--save-path", required=False, default=".\\", help="Path to save trained weights")
    parser.add_argument("--weights", required=False, help="Path of weights, only valid for sepvit and sesepvit, other networks will use the online weights")
    parser.add_argument("--batch-size", required=False, default=256, help="Batch size")
    parser.add_argument("--dataset", required=True, help="Path of dataset")
    parser.add_argument("--data-parallel", action= "store_true", help="Add this if weights to load were trained with multiple GPUs")
    args = parser.parse_args()

    model_name = args.model_name
    save_path = args.save_path
    weights = args.weights
    batch_size = int(args.batch_size)
    dataset_path = args.dataset


    if model_name == "vgg16":
        model = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features, out_features=3)
    elif model_name == "vgg13":
        model = models.vgg13_bn(weights=models.VGG13_BN_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features, out_features=3)
    elif model_name == "vgg19":
        model = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features, out_features=3)
    elif model_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=3)
    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=3)
        model = SeSepViT(
            num_classes = 3,
            dim = 12,               # dimensions of first stage, which doubles every stage (32, 64, 128, 256) for SepViT-Lite
            dim_head = 12,          # attention head dimension
            heads = (1, 2, 2, 4),   # number of heads per stage
            depth = (1, 2, 4, 2),   # number of transformer blocks per stage
            window_size = 7,        # window size of DSS Attention block
            dropout = 0.1           # dropout
        )
    elif model_name == "vit":
        model = timm.create_model('vit_base_patch8_224', pretrained=True)

    if weights is not None:
        checkpoint = torch.load(weights, map_location="cpu")
        model.load_state_dict(checkpoint)
        if model_name == "sesepvit" or model_name == "sepvit":
            model.mlp_head[-1] = nn.Linear(model.dims[-1], 3)

    if model_name[:3] == "vgg":
        model = nn.Sequential(model.features, model.avgpool)
    elif model_name[:6] == "resnet":
        model = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4, model.avgpool)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.data_parallel:
        model = nn.DataParallel(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("Evaluating : " + model_name + "\n")
    print(model)


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(224)
    ])


    patients = os.listdir(dataset_path)
    for patient in patients:
        cur_path = os.path.join(dataset_path, patient)
        dataset = slide_eval_dataset(cur_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=20)
        model.eval()

        features = np.empty(shape=(0, feature_length[model_name]))

        for data, target in dataloader:
            data, target = data.cuda(), target.cuda()
            s_output = model(data)
            if model_name[:3] == "vgg" or model_name[:6] == "resnet":
                s_output = torch.flatten(s_output, 1)
            s_output = s_output.cpu().detach().numpy()
            features = np.concatenate((features, s_output), axis=0)
        
        avg_feature = np.average(features, axis=0)
        std_feature = np.std(features, axis=0)
        np.save(os.path.join(save_path, patient + "avg.npy"), avg_feature)
        np.save(os.path.join(save_path, patient + "std.npy"), std_feature)