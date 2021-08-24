from munch import Munch

from models import LeNet5, AlexNet, VGG16, GoogLeNet


def build_model(args):
    if args.which_model == 'LeNet-5':
        classifier = LeNet5(args)
    elif args.which_model == 'AlexNet':
        classifier = AlexNet(args)
    elif args.which_model == 'VGG16':
        classifier = VGG16(args)
    elif args.which_model == 'GoogLeNet':
        classifier = GoogLeNet(args)
    else:
        raise NotImplementedError

    nets = Munch(classifier=classifier)
    return nets
