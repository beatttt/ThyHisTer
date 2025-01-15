import timm
import torch.nn as nn

# 'vit_base_patch8_224', 'vit_base_patch8_224_in21k', 'vit_base_patch16_224', 
# 'vit_base_patch16_224_in21k', 'vit_base_patch16_224_miil', 'vit_base_patch16_224_miil_in21k',
# 'vit_base_patch16_384', 'vit_base_patch16_sam_224', 'vit_base_patch32_224', 
# 'vit_base_patch32_224_in21k', 'vit_base_patch32_384', 'vit_base_patch32_sam_224',
# 'vit_base_r26_s32_224', 'vit_base_r50_s16_224', 'vit_base_r50_s16_224_in21k',
# 'vit_base_r50_s16_384', 'vit_base_resnet26d_224', 'vit_base_resnet50_224_in21k',
# 'vit_base_resnet50_384', 'vit_base_resnet50d_224', 'vit_giant_patch14_224',
# 'vit_gigantic_patch14_224', 'vit_huge_patch14_224', 'vit_huge_patch14_224_in21k',
# 'vit_large_patch16_224', 'vit_large_patch16_224_in21k', 'vit_large_patch16_384',
# 'vit_large_patch32_224', 'vit_large_patch32_224_in21k', 'vit_large_patch32_384',
# 'vit_large_r50_s32_224', 'vit_large_r50_s32_224_in21k', 'vit_large_r50_s32_384',
# 'vit_small_patch16_224', 'vit_small_patch16_224_in21k', 'vit_small_patch16_384',
# 'vit_small_patch32_224', 'vit_small_patch32_224_in21k', 'vit_small_patch32_384',
# 'vit_small_r26_s32_224', 'vit_small_r26_s32_224_in21k', 'vit_small_r26_s32_384',
# 'vit_small_resnet26d_224', 'vit_small_resnet50d_s16_224', 'vit_tiny_patch16_224',
# 'vit_tiny_patch16_224_in21k', 'vit_tiny_patch16_384', 'vit_tiny_r_s16_p8_224',
# 'vit_tiny_r_s16_p8_224_in21k', 'vit_tiny_r_s16_p8_384'


def create_vit(num_layers=12):
    model = timm.create_model('vit_base_patch8_224', pretrained=True)
    modules = list(model.children())
    sequence = modules[2]
    layers = nn.Sequential()
    for i in range(num_layers):
        layers.append(sequence[i])
    modules[2] = layers
    res = nn.Sequential()
    for i in range(len(modules) - 1):
        if i == 2:
            res.append(layers)
        else:
            res.append(modules[i])
    res.append(nn.Linear(768, 3))
    return res


#  model = create_vit(3)
#  print(model)
