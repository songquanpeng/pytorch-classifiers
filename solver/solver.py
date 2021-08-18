import datetime
import time

import torch
import torch.nn.functional as F
from munch import Munch
from tqdm import tqdm

from data.fetcher import Fetcher
from models.build import build_model
from solver.utils import he_init
from utils.checkpoint import CheckpointIO
from utils.file import write_record, delete_model, delete_sample
from utils.misc import send_message
from utils.model import print_network


class Solver:
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device(args.device)
        self.nets = build_model(args)
        for name, module in self.nets.items():
            print_network(module, name)
        for net in self.nets.values():
            net.to(self.device)

        if args.mode == 'train':
            # Setup optimizers for all nets to learn.
            self.optims = Munch()
            for net in self.nets.keys():
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    weight_decay=args.weight_decay)
            self.ckptios = [
                CheckpointIO(args.model_dir + '/{:06d}_nets.ckpt', **self.nets),
                CheckpointIO(args.model_dir + '/{:06d}_optims.ckpt', **self.optims)]
        else:
            self.ckptios = [CheckpointIO(args.model_dir + '/{:06d}_nets.ckpt', **self.nets)]

        self.use_tensorboard = args.use_tensorboard
        if self.use_tensorboard:
            from utils.logger import Logger
            self.logger = Logger(args.log_dir)

    def initialize_parameters(self):
        if self.args.parameter_init == 'he':
            for name, network in self.nets.items():
                if name not in self.args.pretrained_models:
                    print('Initializing %s...' % name, end=' ')
                    network.apply(he_init)
                    print('Done.')
        elif self.args.parameter_init == 'default':
            # Do nothing because the parameters has been initialized in this manner.
            pass

    def train_mode(self, training=True):
        for nets in [self.nets]:
            for name, network in nets.items():
                # We don't care the pretrained models, they should be set to eval() when loading.
                if name not in self.args.pretrained_models:
                    network.train(mode=training)

    def eval_mode(self):
        self.train_mode(training=False)

    def save_model(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def load_model(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def load_model_from_path(self, path):
        for ckptio in self.ckptios:
            ckptio.load_from_path(path)

    def zero_grad(self):
        for optimizer in self.optims.values():
            optimizer.zero_grad()

    def train(self, loaders):
        args = self.args
        nets = self.nets
        optims = self.optims

        train_fetcher = Fetcher(loaders.train, args)

        # Load or initialize the model parameters.
        if args.start_iter > 0:
            self.load_model(args.start_iter)
        else:
            self.initialize_parameters()

        best_acc = 0
        best_step = 0
        print('Start training...')
        start_time = time.time()
        for step in range(args.start_iter + 1, args.end_iter + 1):
            self.train_mode()
            sample = next(train_fetcher)

            # Train the classifier
            y_hat = nets.classifier(sample.x)
            loss = F.cross_entropy(y_hat, sample.y)
            loss_ref = Munch(loss=loss.item())
            self.zero_grad()
            loss.backward()
            optims.classifier.step()

            self.eval_mode()

            if step % args.log_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "[%s]-[%i/%i]: " % (elapsed, step, args.end_iter)
                all_losses = dict()
                for key, value in loss_ref.items():
                    all_losses[key] = value
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)
                if args.save_loss:
                    if step == args.log_every:
                        header = ','.join(['iter'] + [str(loss) for loss in all_losses.keys()])
                        write_record(header, args.loss_file, False)
                    log = ','.join([str(step)] + [str(loss) for loss in all_losses.values()])
                    write_record(log, args.loss_file, False)
                if self.use_tensorboard:
                    for tag, value in all_losses.items():
                        self.logger.scalar_summary(tag, value, step)

            if step % args.save_every == 0:
                self.save_model(step)
                last_step = step - args.save_every
                if last_step != best_step and not args.keep_all_models:
                    delete_model(args.model_dir, last_step)

            if step % args.eval_every == 0:
                acc = self.evaluate_model(self.nets, loaders.test)
                if acc > best_acc:
                    # New best model existed, delete old best model's weights and samples.
                    if not args.keep_all_models:
                        delete_model(args.model_dir, best_step)
                    best_acc = acc
                    best_step = step
                else:
                    # Otherwise just delete the samples.
                    if not args.keep_all_eval_samples:
                        delete_sample(args.eval_dir, step)
                info = f"step: {step} current acc: {acc * 100:.4f}% history best acc: {best_acc * 100:.4f}%"
                send_message(info, args.exp_id)
                write_record(info, args.record_file)
        send_message("Model training completed.")
        if not args.keep_best_eval_samples:
            delete_sample(args.eval_dir, best_step)

    @torch.no_grad()
    def evaluate_model(self, nets, loader):
        count = 0
        for x, y in tqdm(loader):
            x, y = x.to(self.device), y.to(self.device)
            output = nets.classifier(x)
            y_hat = output.argmax(dim=1)
            count += y_hat.eq(y).sum().item()
            pass
        acc = count / (len(loader) * self.args.batch_size)
        return acc

    @torch.no_grad()
    def evaluate(self, loader):
        acc = self.evaluate_model(self.nets, loader)
        send_message(f"ACC: {acc * 100:.4f}%")
