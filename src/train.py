from __future__ import print_function
import os
import torch
from components.dataset import patch_dataset
from torchvision import transforms
import torch.nn as nn
from torch.optim import lr_scheduler
import time
import copy
from components.builder import build
from components.se_sep_transformer import FuSepViT
from torchvision import models
from components.fusion import fusion


# python .\train.py --model-name densenet161 --dataset E:\patch\ --save-path ..\finetuned_weights --batch-size 512 --data-parallel
# 跑融合模型时：
# python .\train.py --model-name fusion --dataset E:\patch\ --save-path ..\finetuned_weights --batch-size 512 --data-parallel
if __name__ == "__main__":
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

    # 需要自己预训练的模型
    if args.model_name in ["seresnet18", "seresnet20", "seresnet34", "seresnet50", "seresnet56", "seresnet101", "seresnet152", 
                           "sepvit_lite", "sepvit_tiny", "sepvit_small", "sepvit_base", "sesepvit_lite_s", "sesepvit_lite", 
                           "sesepvit_tiny", "sesepvit_small", "sesepvit_base"]:
        print("Loading pretrained weights...")
        checkpoint = torch.load(args.pretrained_weights, map_location="cpu")["state_dict"]

        if list(checkpoint.keys())[0].split(".")[0] == "module":
            for k in list(checkpoint):
                tmp = "."
                checkpoint[tmp.join(k.split(".")[1:])] = checkpoint.pop(k)

        model.load_state_dict(checkpoint)

    if args.breakpoint is not None:
        print("Loading breakpoint...")
        checkpoint = torch.load(args.breakpoint, map_location="cpu")

        if list(checkpoint.keys())[0].split(".")[0] == "module":
            for k in list(checkpoint):
                tmp = "."
                checkpoint[tmp.join(k.split(".")[1:])] = checkpoint.pop(k)

        model.load_state_dict(checkpoint)


    transform_train = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomRotation(15),
         transforms.ColorJitter(),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip(),
         transforms.ToTensor()
         ])
    transform_validation = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor()])
    data_transforms = {'train': transform_train,
                       'validation': transform_validation}


    datasets = {x: patch_dataset(os.path.join(args.dataset, x), data_transforms[x]) for x in ['train', 'validation']}
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=int(args.batch_size), shuffle=True, num_workers=20) for x in ['train', 'validation']}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'validation']}
    optimizer = torch.optim.SGD([{'params': model.parameters()}], lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


    print("Training : ")
    print(model)
    device = torch.device(args.device)
    if args.data_parallel:
        print("Using " + str(torch.cuda.device_count()) + "GPUs")
        model = nn.DataParallel(model)
    model = model.to(args.device)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1, args.epoch + 1):
        torch.cuda.empty_cache()
        print('Epoch {}/{}'.format(epoch, args.epoch))
        print('-' * 10)
        # Each epoch has a training and validationidation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        torch.save(best_model_wts, args.model_name + "_" + str(epoch) + ".pth")
        if os.path.exists(args.model_name + "_" + str(epoch - 1) + ".pth"):
            os.remove(args.model_name + "_" + str(epoch - 1) + ".pth")
        print()

    if os.path.exists(args.model_name + "_" + str(args.epoch) + ".pth"):
        os.remove(args.model_name + "_" + str(args.epoch) + ".pth")
    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation Acc: {:4f}'.format(best_acc))
    torch.save(best_model_wts, os.path.join(args.save_path, args.model_name + '.pth'))