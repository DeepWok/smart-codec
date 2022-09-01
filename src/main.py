import torch
import numpy as np
import random
import argparse

from torch.utils.data import DataLoader
# from pytorch_lightning.plugins import DDPPlugin
# from pytorch_lightning.strategies.ddp import DDPStrategy


from sessions import train, test
from datasets import get_dataset, get_transform, dataset_info
from models import get_network_by_name

def main():
    parser = argparse.ArgumentParser(description='Investigate how feature augmentation would help model training')

    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--model', default='simplenet', type=str, help='model type')
    parser.add_argument('--dataset', default='hamming', type=str, help='dataset name')
    parser.add_argument('--dataset_path', default='data/hamming', type=str, help='dataset path')
    parser.add_argument('--optimizer', default='adam', type=str, help='optimizer style')

    parser.add_argument('--save', '-save', default="./test", type=str, help='saving dir')
    parser.add_argument('--load', '-load', default=None, type=str, help='resume from checkpoint')
    parser.add_argument('--eval', '-eval', action='store_true',help='only eval')

    parser.add_argument('--max_epochs', '-m', default=200, type=int, help='max epochs')
    parser.add_argument('--batch_size', '-b', default=256, type=int, help='batch size')

    parser.add_argument('--num_workers', '-w', default=0, type=int, help='number of workers for the dataset')
    parser.add_argument('--num_devices', '-n', default=None, type=int, help='number of devices')
    parser.add_argument('--accelerator', '-a', default=None, type=str, help='accelerator style')
    parser.add_argument('--strategy', '-s', default='ddp', type=str, help='accelerator strategy')

    parser.add_argument('--debug', '-debug', action='store_true',help='debug')
    parser.add_argument('--seed', '-seed', default=0, type=int ,help='seed')


    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # seeding
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


    # get data and transforms
    train_transform, test_transform = get_transform(name=args.dataset)

    train_dataset = get_dataset(
        name=args.dataset, my_path=args.dataset_path, mode='train', transform=train_transform)
    val_dataset = get_dataset(
        name=args.dataset, my_path=args.dataset_path, mode='val', transform=test_transform)      
    test_dataset = get_dataset(
        name=args.dataset, my_path=args.dataset_path, mode='test', transform=test_transform)      


    # get data loader
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # build model
    print(f"Building model {args.model}")
    num_cls = dataset_info[args.dataset]['num_classes']
    model = get_network_by_name(args.model, num_class=num_cls)

    if args.eval:
        trainer_args = {
            'devices': args.num_devices, 
            'accelerator': args.accelerator,
            'strategy': args.strategy,
            # 'plugins':  DDPPlugin(find_unused_parameters=False),
        }
        test(
            model=model,
            test_loader=test_loader,
            trainer_args=trainer_args,
            load_path=args.load)
    else:
        # if args.strategy == 'ddp':
        #     strategy = DDPStrategy(find_unused_parameters=False)
        # else:
        #     strategy = args.strategy

        trainer_args = {
            'max_epochs': args.max_epochs,
            # 'optimizer': args.optimizer,
            'devices': args.num_devices, 
            'accelerator': args.accelerator,
            'strategy': args.strategy,
            'fast_dev_run': args.debug,
            # 'plugins':  DDPPlugin(find_unused_parameters=False),
        }

        train(
            model=model, 
            train_loader=train_loader, 
            val_loader=test_loader, 
            learning_rate=args.lr, 
            trainer_args=trainer_args,
            save_path=args.save,
            optimizer=args.optimizer)


if __name__ == '__main__':
    main()
