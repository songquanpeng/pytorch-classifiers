from munch import Munch
from config import setup_cfg, validate_cfg, load_cfg, save_cfg, print_cfg
from solver.solver import Solver
from data.loader import get_train_loader, get_test_loader


def main(args):
    solver = Solver(args)
    if args.mode == 'train':
        loaders = Munch(train=get_train_loader(**args), test=get_test_loader(**args))
        solver.train(loaders)
    elif args.mode == 'eval':
        solver.evaluate(loader=get_test_loader(**args))
    else:
        assert False, f"Unimplemented mode: {args.mode}"


if __name__ == '__main__':
    cfg = load_cfg()
    setup_cfg(cfg)
    validate_cfg(cfg)
    save_cfg(cfg)
    print_cfg(cfg)
    main(cfg)
