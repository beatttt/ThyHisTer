import argparse
import os
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import GradCAM, \
    HiResCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    GradCAMElementWise


from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from components.builder import build
from components.se_sep_transformer import FuSepViT
from components.fusion import fusion

# 对于不同的ViT需要将height和width设置成不同的值，可以看尺寸不匹配的报错，height * width + 1需要等于特征图尺寸
def reshape_transform(tensor, height=28, width=28):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


# python .\cam.py --model-name sepvit_lite --image-path ..\t_PTC_32_2575.png
if __name__ == '__main__':

    model, args = build()

    if args.model_name == "fusion":
        model1 = FuSepViT(                                                               
            num_classes = 1000,
            dim = 32,           
            dim_head = 32,          
            heads = (1, 2, 4, 8),  
            depth = (1, 2, 6, 2),  
            window_size = 7,       
            dropout = 0.1          
        )
        print("Loading pretrained weights...")
        checkpoint = torch.load("..\\pretrained_weights\\sepvit_lite.pth", map_location="cpu")["state_dict"]
        if list(checkpoint.keys())[0].split(".")[0] == "module":
            for k in list(checkpoint):
                tmp = "."
                checkpoint[tmp.join(k.split(".")[1:])] = checkpoint.pop(k)
        checkpoint.pop("mlp_head.1.weight")
        checkpoint.pop("mlp_head.1.bias")
        checkpoint.pop("mlp_head.2.weight")
        checkpoint.pop("mlp_head.2.bias")
        model1.load_state_dict(checkpoint)


        model2 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        layers = list(model2.children())[:-2]
        model2 = torch.nn.Sequential(*layers)

        model = fusion(model1, model2, 768, 3)

    model.eval()
    if args.weights is not None:
        checkpoint = torch.load(args.weights, map_location="cpu")

        if list(checkpoint.keys())[0].split(".")[0] == "module":
            for k in list(checkpoint):
                tmp = "."
                checkpoint[tmp.join(k.split(".")[1:])] = checkpoint.pop(k)

        model.load_state_dict(checkpoint)

    if args.model_name == "alexnet" or args.model_name[:3] == "vgg" or args.model_name[:8] == "densenet":
        target_layers = [model.features]
    elif args.model_name[:6] == "resnet":
        target_layers = [model.layer4]
    elif args.model_name[:8] == "vit":
        target_layers = [model.blocks[-1].norm1]
    elif args.model_name[:8] == "sesepvit" or args.model_name[:6] == "sepvit":
        target_layers = [model.layers[-1][-1]]
    elif args.model_name == "fusion":
        target_layers = [model.fusion_block]

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [e.g ClassifierOutputTarget(281)]
    targets = None

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = GradCAM
    with cam_algorithm(model=model,
                       target_layers=target_layers,
                       use_cuda=args.device == "cuda",
                       reshape_transform=reshape_transform if args.model_name == "vit" else None) as cam:
        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=False,
                            eigen_smooth=False)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.device == "cuda")
    gb = gb_model(input_tensor, target_category=None)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    cv2.imwrite(os.path.join(args.save_path, args.model_name + "_cam.jpg"), cam_image)
    # cv2.imwrite(os.path.join(args.save_path, args.model_name + "_gb.jpg"), gb)
    # cv2.imwrite(os.path.join(args.save_path, args.model_name + "_cam_gb.jpg"), cam_gb)
