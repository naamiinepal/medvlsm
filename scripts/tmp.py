import torch 
import torchvision
import os
import clip

# Suppose you are trying to load pre-trained resnet model in directory- models\resnet

#os.environ['TORCH_HOME'] = 'models\\resnet' #setting the environment variable
#resnet = torchvision.models.resnet50(pretrained=True)


print(clip.available_models())
clip.load('RN50', device="cuda", jit=False)
