# -*- coding: utf-8 -*-

import argparse
from time import sleep

args = None

def main(arguments):
    global args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--param', type=str, default="12",
                        help='Hyper parameter, can also be a list')
    parser.add_argument('--L', type=float, default=0.1,
                        help='Hyper parameter for SGD')
    args = parser.parse_args(arguments)
    args.param = eval(args.param)
    args.param = args.param if type(args.param) == list else [args.param]
    return train()

def train():
    sleep(10)
    return (args.param[0], args.L)

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
