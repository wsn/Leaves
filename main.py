import argparse
import os
import pdb
import tensorflow as tf

from Solvers import create_solver
from options import parse_opt


def main():
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    tf.enable_eager_execution(config=config)

    parser = argparse.ArgumentParser()
    parser.add_argument('--options', type=str, help='Path to the option JSON file.')
    args = parser.parse_args()
    opt = parse_opt(args.options)
    
    solver = create_solver(opt)
    
    if opt['is_training']:
        solver.train()
    else:
        solver.test()


if __name__ == '__main__':
    main()
