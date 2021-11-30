import torch
from torch import nn
import os
from models import densenet, resnet, vanilla_cifar_resnet
from models.efficientnet import EfficientNet


def get_classifier(opt):
    pretrained = opt.get("pretrained", "")
    pretrained_fc = opt.get("pretrained_fc", None)
    if os.path.isfile(pretrained) and pretrained_fc > 2 and type(pretrained_fc) == int:
        pretrained_flag = True
        num_classes = pretrained_fc
        ckpt = torch.load(pretrained)
        print("pretrained model {} with {} classes found.".format(pretrained, pretrained_fc))
    else:
        pretrained_flag = False
        num_classes = opt['num_classes']

    if "densenet" in opt['arch']:
        depth = opt['model_depth']
        if depth == 121:
            model = densenet.densenet121(num_classes=num_classes)
        elif depth == 161:
            model = densenet.densenet161(num_classes=num_classes)
        elif depth == 169:
            model = densenet.densenet169(num_classes=num_classes)
        elif depth == 201:
            model = densenet.densenet201(num_classes=num_classes)
        else:
            raise ValueError("Invalid value {} of depth for densenet arch".format(depth))
    elif "resnet" == opt['arch']:
        assert opt['model_depth'] in [10, 18, 34, 50, 101, 152, 200]
        if opt['model_depth'] == 18:
            model = resnet.resnet18(
                num_classes=309,
                pool=opt['pool'])
            # model.load_state_dict(torch.load("resnet18_weight.pth"))
            fc_in = model.fc.in_features
            model.fc = nn.Linear(fc_in, num_classes)
        elif opt['model_depth'] == 34:
            model = resnet.resnet34(
                num_classes=num_classes,
                pool=opt['pool'])
        elif opt['model_depth'] == 50:
            model = resnet.resnet50(
                num_classes=num_classes,
                pool=opt['pool'])
        elif opt['model_depth'] == 101:
            model = resnet.resnet101(
                num_classes=num_classes)
        elif opt['model_depth'] == 152:
            model = resnet.resnet152(
                num_classes=num_classes)
    elif "cifar_resnet" == opt['arch']:
        depth = opt['model_depth']
        if depth == 20:
            model = vanilla_cifar_resnet.resnet20(num_classes=num_classes)
        elif depth == 32:
            model = vanilla_cifar_resnet.resnet32(num_classes=num_classes)
        elif depth == 34:
            model = vanilla_cifar_resnet.resnet34_custom(num_classes=num_classes)
        elif depth == 44:
            model = vanilla_cifar_resnet.resnet44(num_classes=num_classes)
        elif depth == 56:
            model = vanilla_cifar_resnet.resnet56(num_classes=num_classes)
        elif depth == 110:
            model = vanilla_cifar_resnet.resnet110(num_classes=num_classes)
        else:
            raise ValueError("Invalid value {} of depth for cifar_resnet arch".format(depth))
    elif "efficientnet" == opt['arch']:
        im_size = opt.get("im_size", 221)
        model_config = opt['model_depth']
        model = EfficientNet.from_name("{}-{}".format(opt['arch'], model_config), in_channels=1,
                                       image_size=None, num_classes=num_classes)
    else:
        raise ValueError("Unsupported value {} for opt['arch']".format(opt['arch']))
    if pretrained_flag:
        if "resnet" == opt['arch']:
            fc_in = model.fc.in_features
            print("pretrained loading: ", model.load_state_dict(ckpt))
            model.fc = nn.Linear(fc_in, opt['num_classes'])
        elif "densenet" == opt['arch']:
            fc_in = model.classifier.in_features
            print("pretrained loading: ", model.load_state_dict(ckpt))
            model.classifier = nn.Linear(fc_in, opt['num_classes'])
        elif "cifar_resnet" == opt['arch']:
            fc_in = model.linear.in_features
            print("pretrained loading: ", model.load_state_dict(ckpt))
            model.linear = nn.Linear(fc_in, opt['num_classes'])
        elif "efficientnet" == opt['arch']:
            fc_in = model._fc.in_features
            print("pretrained loading: ", model.load_state_dict(ckpt))
            model._fc = nn.Linear(fc_in, opt['num_classes'])
    return model