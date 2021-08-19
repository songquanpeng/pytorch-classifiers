from munch import Munch

from models import LeNet5, AlexNet


def build_model(args):
    if args.which_model == 'LeNet-5':
        classifier = LeNet5(args)
    elif args.which_model == 'AlexNet':
        classifier = AlexNet(args)
    else:
        raise NotImplementedError

    nets = Munch(classifier=classifier)
    return nets
