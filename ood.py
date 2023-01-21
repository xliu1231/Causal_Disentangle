import argparse
import os
import copy
import torch
import torch.nn as nn
from torchvision import transforms as T
from torch.utils.data import random_split
import datetime
from torch.utils.data import Dataset, DataLoader
from vae import VAE
from causalvae import CausalVAE, CausalDAG
from datasets import ShapeDataset
import torch.optim as optim


def parse():
    parser = argparse.ArgumentParser(description="ood evaluation")

    # data
    parser.add_argument("--dataset", default="3dshape", type=str,
                        help="The dataset used for training.")
    parser.add_argument("--image_height", default=64, type=int,
                        help="The image height of each sample.")
    parser.add_argument("--image_width", default=64, type=int,
                        help="The image width  of each sample.")
    parser.add_argument("--image_channels", default=3, type=int,
                        help="The number of channels in each sample.")
    parser.add_argument("--ind_data_path", default=None, type=str,
                        help="The path to the folder stroing the data.")
    parser.add_argument("--ind_attr_path", default=None, type=str,
                        help="The path to the folder stroing the attribute of the data.")
    parser.add_argument("--ood_data_path", default=None, type=str,
                        help="The path to the folder stroing the data.")
    parser.add_argument("--ood_attr_path", default=None, type=str,
                        help="The path to the folder stroing the attribute of the data.")
    parser.add_argument("--label_idx", default=5, type=int,
                        choices=[0, 1, 2, 3, 4, 5],
                        help="Use which attribute as the label. The index of the attribute in "
                             "['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']")

    # model
    # todo: model type and hyperparams of the model
    parser.add_argument("--model_type", default="vae", type=str, choices=['vae', 'causalvae'],
                        help="The type of the model")
    parser.add_argument('--z_dim', default=128, type=int)
    parser.add_argument('--num_concepts', default=4, type=int)
    parser.add_argument('--dim_per_concept', default=32, type=int)
    parser.add_argument("--model_path", default=None, type=str,
                        help="Path of the trained model.")

    # training
    parser.add_argument("--batch_size", default=128, type=int, metavar='B',
                        help="The batch size for training.")
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--epoch_num', default=100, type=int)
    parser.add_argument("--train_ratio", default=0.9, type=float,
                        help="The ratio of training samples in the .")
    parser.add_argument("--val_epoch", default=10, type=int,
                        help="Log the learning curve every print_freq iterations.")

    # optimization
    parser.add_argument('--opt_type', type=str, default='sgd', choices=['sgd', 'adam'],
                        help='type of optimizer')
    parser.add_argument('--base_lr', type=float, default=0.1)
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine',
                        choices=['multistep', 'cosine'])
    parser.add_argument('--lr_milestones', nargs='+', type=int, default=[30, 60, 90, 100],
                        help="used with multistep lr_scheduler")

    # ood setting
    parser.add_argument('--train_from_scratch', action='store_true', default=False,
                        help='If true, use no pre-trained encoder.')
    parser.add_argument('--freeze_encoder', action='store_true', default=False,
                        help='If true, freeze the encoder')
    parser.add_argument('--eval_only', action='store_true', default=False,
                        help='If true, evaluate classifier without training')  # todo: haven't done
    parser.add_argument('--eval_model_path', type=str, default=None,
                        help='Path of the trained classifier.')

    # others
    parser.add_argument("--seed", type=int, default=1, metavar='S', help="random seed (default: 1)")
    parser.add_argument("--save_path", default=None, type=str,
                        help="The path to save model.")
    parser.add_argument("--save_name", default="", type=str,
                        help="The name of the saved model.")

    return parser.parse_args()


def get_num_class(idx, dataset):
    if dataset == '3dshape':
        _FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
        _NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 'scale': 8,
                                  'shape': 4,
                                  'orientation': 15}
        num_class = _NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[idx]]
    else:
        raise Exception(f'Unknown dataset {args.dataset}')
    return num_class


def load_data(args):
    if args.dataset == '3dshape':
        transform = T.ToTensor()
        ind_set = ShapeDataset(data_path=args.ind_data_path, attr_path=args.ind_attr_path,
                               attr=args.label, transform=transform)  # todo: attr list?
        len_train = int(len(ind_set) * args.ratio)
        ind_train_set, ind_test_set = random_split(ind_set, [len_train, len(ind_set) - len_train],
                                                   generator=torch.Generator().manual_seed(
                                                       args.seed))

        ood_test_set = ShapeDataset(data_path=args.ood_data_path, attr_path=args.ood_attr_path,
                                    attr=args.label, transform=transform)  # todo: attr list?

    else:
        raise Exception(f'Unknown dataset {args.dataset}')
    return ind_train_set, ind_test_set, ood_test_set


def resume(model, model_path):
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        print("Model loaded")
    else:
        print("=> no checkpoint found at '{}'".format(model_path))
    return model


class Classifier(nn.Module):
    def __init__(self, encoder: nn.Module, z_dim: int, num_class: int):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.z_dim = z_dim
        self.num_class = num_class
        self.classifier = nn.Linear(self.z_dim, self.num_class)

    def forward(self, x):
        result = self.encoder(x)
        mu = result[:, :self.z_dim]  # todo: use result to be representation?
        out = self.classifier(mu)
        return out


def load_optimizer(model, args):
    # freeze encoder or not
    if args.freeze_encoder:
        params = model.classifier.parameters()
    else:
        params = model.parameters()

    if args.opt_type == 'sgd':
        optimizer = optim.SGD(params, lr=args.base_lr, weight_decay=5e-4, momentum=0.9)
    elif args.opt_type == 'adam':
        optimizer = optim.Adam(params, lr=args.base_lr)
    else:
        raise Exception(f'Unknown optimizer {args.opt_type}')

    if args.lr_scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch_num)
    elif args.lr_scheduler_type == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.lr_milestones)
    else:
        raise Exception(f'Unknown lr scheduler {args.lr_scheduler_type}')
    return optimizer, scheduler


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, save_path, filename):
    filename += "_checkpoint.pth.tar"
    filename = save_path + '/' + filename
    torch.save(state, filename)


def accuracy(output, target, topk=(1,), exact=False):
    with torch.no_grad():
        # Binary Classification
        if len(target.shape) > 1:
            assert output.shape == target.shape, \
                "Detected binary classification but output shape != target shape"
            return [torch.round(torch.sigmoid(output)).eq(torch.round(target)).float().mean()], [
                -1.0]

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        res_exact = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float()
            ck_sum = correct_k.sum(0, keepdim=True)
            res.append(ck_sum.mul_(100.0 / batch_size))
            res_exact.append(correct_k)

        if not exact:
            return res
        else:
            return res_exact


def eval(args, val_loader, model):
    model.eval()
    accs = AverageMeter()
    with torch.no_grad():
        for _, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            outputs = model(inputs)
            prec = accuracy(outputs, labels)[0]
            accs.update(prec.item(), inputs.size(0))
    return accs.avg


def main(args):
    # load data
    ind_train_set, ind_test_set, ood_test_set = load_data(args)
    print('In-distribution training set size:', len(ind_train_set))
    print('In-distribution test set size:', len(ind_test_set))
    print('Out-of-distribution test set size:', len(ood_test_set))

    train_loader = DataLoader(ind_train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers)
    test_loader = DataLoader(ind_test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers)
    ood_loader = DataLoader(ood_test_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers)

    # make model
    num_class = get_num_class(args.label_idx)
    if args.model_type == 'vae':
        unsup_model = VAE(z_dim=args.z_dim)
    elif args.model_type == 'causalvae':
        unsup_model = CausalVAE(args.z_dim, args.num_concepts, args.dim_per_concept)

    # load pre-trained model
    if not args.train_from_scratch:
        unsup_model = resume(unsup_model, args.model_path)

    # define classifier
    encoder = unsup_model.encoder
    model = Classifier(encoder, args.z_dim, num_class)
    model = model.to(args.device)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = load_optimizer(model, args)

    # train classifier
    best_val_acc = -1
    for epoch in range(args.epoch_num):
        losses = AverageMeter()
        model.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # get data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            model.train()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # update results
            losses.update(loss.item(), inputs.size(0))
        train_acc = eval(args, ind_train_set, model)
        now = datetime.datetime.now()
        print(
            '[{}] Epoch {:2d} | Loss {:.4f} | Acc {:.2f}%'.format(now.strftime('%Y-%m-%d %H:%M:%S'),
                                                                  epoch, losses.avg, train_acc))

        if epoch % args.val_epoch == 0 or epoch == args.epoch_num - 1:
            # validation
            ind_acc = eval(args, test_loader, model)
            ood_acc = eval(args, ood_loader, model)
            print('Val Epoch {:2d} | Ind Acc {:.2f}% | Ood Acc {:.2f}%'.format(epoch, ind_acc,
                                                                               ood_acc))
            if ind_acc > best_val_acc:
                best_val_acc = ind_acc
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
        scheduler.step()
    print('Finish training!')

    # save model
    save_checkpoint({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, args.save_path, args.save_name)

    # test
    model.load_state_dict(best_model_state)
    ind_acc = eval(args, test_loader, model)
    ood_acc = eval(args, ood_loader, model)
    print(
        'Test: Ind Acc is {:.2f}%, Ood Acc is {:.2f}%, best epoch is {:2d}'.format(ind_acc, ood_acc,
                                                                                   best_epoch))

    return


if __name__ == "__main__":
    args = parse()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(args)
    main(args)
