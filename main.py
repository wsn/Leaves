import argparse
import os
import pdb
import tensorflow as tf
import numpy as np

from Solvers import create_solver
from options import parse_opt


def main():
    
    # Switch Eager Execution Mode.
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    tf.enable_eager_execution(config=config)
    
    # Deterministic Settings.
    tf.set_random_seed(1)
    np.random.seed(1)
    
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--options', type=str, help='Path to the option JSON file.')
    args = parser.parse_args()
    opt = parse_opt(args.options)
    
    # Create solver.
    solver = create_solver(opt)
    
    # Run solver.
    if opt['is_training'] == 'train':
        solver.train()
    elif opt['is_training'] == 'test':
        solver.test()
    elif opt['is_training'] == 'play':
        solver.play()
    else:
        raise NotImplementedError('Unsupported usage.')


if __name__ == '__main__':
    main()
