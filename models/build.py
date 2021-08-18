from munch import Munch

from models import LeNet5


def build_model(args):
    if args.which_model == 'LeNet-5':
        classifier = LeNet5(args)
    else:
        raise NotImplementedError

    nets = Munch(classifier=classifier)
    return nets
