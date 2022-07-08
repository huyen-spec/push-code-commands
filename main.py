import config as CFG
from detectron2.engine import launch



# python main.py --name baseline --batch_size 4 --mode test --n_gpus 1 --resume_path model_0019999.pth


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def register_dataset():
    '''
    Register a pill dataset.
    '''
    from detectron2.data.datasets import register_coco_instances

    register_coco_instances("pills_train", {}, "data/pills/data_train/instances_train.json", "")
    register_coco_instances("pills_test", {}, "data/pills/data_test/instances_test.json", "")

def main(args):
    register_dataset()
    if args.name.startswith('KGP_e2e'):
        from models.KGPNet_e2e import train, test
    elif args.name.startswith('KGP_2step'):
        from models.KGPNet_2step import train, test
    elif args.name.startswith('baseline'):
        from models.faster_rcnn import train, test
    elif args.name.startswith('sgrn'):
        from models.faster_rcnn_sgrn import train, test
    else:
        raise NotImplementedError

    print("CFG***********", CFG)

    if args.mode == 'train':
        seed_everything(CFG.seed)
        train(args)
    else:
        seed_everything(CFG.seed)
        test(args)
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Pill Detection')

    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA training')
    parser.add_argument('--seed', type=int, default=911, metavar='S',
                        help='random seed (default: 911)')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N', dest='batch_size')
    parser.add_argument('--val_batch_size', type=int, default=16, metavar='N', dest='v_batch_size')
    parser.add_argument('--name', type=str, default="baseline", metavar='N',
                        help='name of saving model')
    parser.add_argument('--patience', type=int, default=20, metavar='N',
                        help='patience of early stopping')
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
                        help='number of training epochs')
    parser.add_argument('--backbone', type=str, default="resnet50", metavar='N', help='choose backbone model')
    parser.add_argument('--mode', type=str, default="train", metavar='N', help='train or test')
    parser.add_argument('--lr', type=float, default=0.001, metavar='N', help='learning rate')
    parser.add_argument('--n_workers', type=int, default=CFG.n_workers, metavar='N', help='number of workers')
    parser.add_argument('--resume', type=bool, default=False, metavar='N', help='resume training')
    parser.add_argument('--resume_path', type=str, default='', metavar='N', help='resume training/testing path')
    parser.add_argument('--n_classes', type=int, default=CFG.n_classes, metavar='N', help='number of classes')
    parser.add_argument('--max_iters', type=int, default=CFG.max_iters, metavar='N', help='number of max iteration')
    parser.add_argument('--linking_loss_weight', type=float, default=CFG.linking_loss_weight, metavar='N', help='weight of linking loss')
    parser.add_argument('--train_gcn', type=bool, default=False, metavar='N')
    parser.add_argument('--n_gpus', type=int, default=1, metavar='N', help='number of gpus used for training')
    parser.add_argument('--save_best', type=bool, default=True, metavar='N', help='save best after specified epochs')
    
    args = parser.parse_args()
    print('args************')
    print(args)
    # init_distributed_mode(args)
    # seed_everything(args.seed)
    launch(main, args.n_gpus, args=(args,))