import argparse
import os
import pdb
import tensorflow as tf

from Solvers import create_solver
from options import parse_opt


def main():
    
    # Use TensorFlow 2.0 Features.
    tf.enable_v2_behavior()
    
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--options', type=str, help='Path to the option JSON file.')
    args = parser.parse_args()
    opt = parse_opt(args.options)
    
    # Create solver.
    solver = create_solver(opt)
    
    # Run solver.
    if opt['is_training']:
        solver.train()
    else:
        solver.test()


if __name__ == '__main__':
    main()
