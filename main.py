import argparse, os
from trainer import Trainer

if __name__ == '__main__':
    """ argparsers """
    parser = argparse.ArgumentParser(description='Fairness Through Matching')
    
    parser.add_argument('--alg', type=str, default='FTM', choices=['FTM', 'unfair'])
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--model_dir', type=str, default=f'{os.getcwd()}/models/')
    parser.add_argument('--result_dir', type=str, default=f'{os.getcwd()}/results/')
    parser.add_argument('--source', type=str, default='Adult_0')
    parser.add_argument('--target', type=str, default='Adult_1')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--model_lr', type=float, default=1e-3)
    parser.add_argument('--lmda_f', type=float, default=0.5)
    parser.add_argument('--eval_env', default='test')

    args = parser.parse_args()

    print('\t' + '='*30)
    for key, value in vars(args).items():
        print(f'\t [{key}]: {value}')

    trainer_module = Trainer(args)
    trainer_module.load_data(args)
    trainer_module.train_f(args)