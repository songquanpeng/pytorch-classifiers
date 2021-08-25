from munch import Munch
import models
from models import *


def build_model(args):
    if args.which_model == 'LeNet-5':
        classifier = LeNet5(args)
    elif args.which_model == 'AlexNet':
        classifier = AlexNet(args)
    elif args.which_model == 'VGG16':
        classifier = VGG16(args)
    elif args.which_model == 'GoogLeNet':
        classifier = GoogLeNet(args)
    elif args.which_model.startswith('ResNet'):
        model_name = 'ResNet' + args.which_model[7:]
        classifier = getattr(models, model_name)(args)
    else:
        raise NotImplementedError

    nets = Munch(classifier=classifier)
    return nets
