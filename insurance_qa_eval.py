from __future__ import print_function

import os

import sys
import random
from time import strftime, gmtime, time

import pickle
import json

from scipy.stats import rankdata
from evaluator import Evaluator
import click
import importlib

def log(string):
    print(string)

@click.command()
@click.option('--config', required=True, help='location of the config .py file')
def train(config):
    conf = importlib.import_module(config).conf
    print(conf)
    evaluator = Evaluator(conf, model=conf['model'], optimizer='adam')
    # train the model
    best_loss = evaluator.train()
    # evaluate mrr for a particular epoch
    evaluator.load_epoch(best_loss['epoch'])
    top1, mrr = evaluator.get_score(verbose=False)
    log(' - Top-1 Precision:')
    log('   - %.3f on test 1' % top1[0])
    log('   - %.3f on test 2' % top1[1])
    log('   - %.3f on dev' % top1[2])
    log(' - MRR:')
    log('   - %.3f on test 1' % mrr[0])
    log('   - %.3f on test 2' % mrr[1])
    log('   - %.3f on dev' % mrr[2])


if __name__ == '__main__':
    train()

