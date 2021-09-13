import torch.nn as nn
import torch
from backbones import modified_resnet
from .baselien_conv import ConvAngularPen
from .resnet import (
    Resnet18Triplet,
    Resnet34Triplet,
    Resnet50Triplet
)

from utils_resnet import resnet34, resnet_face18, resnet50


# bypass layer
class Identity(nn.Module):
    def __init__(self, n_inputs):
        super(Identity, self).__init__()
        self.in_features = n_inputs

    def forward(self, x):
        return x


class FeaturesModel(nn.Module):
    def __init__(self, pretrained_model):
        super(FeaturesModel, self).__init__()
        self.pretrained_model = pretrained_model

        #print(*list(pretrained_model.children())[-1])

        self.features = nn.Sequential(
            *list(pretrained_model.children())[:-1])

        #self.features = pretrained_model.module.features

    def forward(self, x):
        x = self.features(x)
        return x



# def get_backbone_model(flags):
#     backbone_name = flags.backbone

#     model = None

#     pretrained = True

#     if flags.freeze:
#         pretrained = True

#     if 'resnet' in backbone_name:
#         if 'resnet_modified' in backbone_name:
#             nb_layers = int(backbone_name.split('_')[-1])
#             model = modified_resnet.resnet18(dropout=flags.dropout,nb_layers=nb_layers,
#                                              pretrained=pretrained)
#         else:
#             model = torch.hub.load('pytorch/vision:v0.6.0',
#                                     backbone_name.split('_')[-1],
#                                     pretrained=False)
            
#             if flags.dropout > 0:
#                 if 'resnet18' in backbone_name or 'resnet34' in backbone_name:
#                     model.fc = nn.Sequential(nn.Dropout(flags.dropout),
#                                              nn.Linear(512, flags.num_classes))
#                 else:
#                     model.fc = nn.Sequential(nn.Dropout(flags.dropout),
#                                              nn.Linear(2048, flags.num_classes))
#             else:
#                 if 'resnet18' in backbone_name or 'resnet34' in backbone_name:
#                     model.fc = nn.Linear(512, flags.num_classes)
#                 else:
#                     model.fc = nn.Linear(2048, flags.num_classes)
#     elif backbone_name == 'convnet':
#         model = ConvAngularPen(num_classes = flags.num_classes, loss_type=flags.loss_type)
#     else:
#         raise  ValueError('No Model found --> {}'.format(backbone_name))

#     return model

def get_backbone_model(flags):
    model_architecture = flags.backbone
    embedding_dimension = flags.embedding_dimension
    pretrained = True
    model = None

    if model_architecture == "resnet18_triplet":
        model = Resnet18Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet34_triplet":
        model = Resnet34Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet50_triplet":
        model = Resnet50Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    
    elif model_architecture == 'resnet18':
        model = resnet_face18(use_se=flags.use_se)
    elif model_architecture == 'resnet34':
        model = resnet34()
    elif model_architecture == 'resnet50':
        model = resnet50()

    print("Using {} model architecture.".format(model_architecture))

    return model