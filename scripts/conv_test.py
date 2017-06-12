import os,sys
sys.path.append(os.environ.get('PKR'))
import argparse
import math
import re
import numpy as np
from skimage import io
from keras import backend as K
from keras.optimizers import Adam 
import models.aframe as af
from helpers.planet import PlanetData
from helpers.dfgen import DFGen
import utils

#
# CONFIG
#
VALID_PCT=0.20
MODEL=af.Flex
DEFAULT_BATCH_NORM=True
INITIAL_LR=0.02
METRICS=['accuracy',utils.k_f2]
FC_LAYERS=[256,512]
""" PRERUN """
PRERUN_LR=0.001
PRERUN_TRAIN_SIZE=500
PRERUN_VALID_SIZE=100
PRERUN_BATCH_SIZE=64
PRERUN_EPOCHS=1
""" DEFAULT_CONFIG """
DEFUALT_TEST_MODE=False
DEFAULT_BATCH_SIZE=64
DEFAULT_EPOCHS=50
""" TEST CONFIG """
TEST_PD_SIZE=300
TEST_TRAIN_SIZE=128
TEST_VALID_SIZE=64
TEST_BATCH_SIZE=32
TEST_EPOCHS=2



#
# MODEL
#
def run(
        data,
        conv_layers,
        batch_size,
        train_size,
        valid_size,
        batch_norm,
        epochs,
        weights,
        test_run):
    ident=_ident(conv_layers)
    print('conv_layers:',conv_layers)
    model_obj=MODEL(
        batch_norm=batch_norm,
        conv_layers=conv_layers,
        fc_layers=FC_LAYERS,
        metrics=METRICS,
        auto_compile=False)
    if test_run:
        names=f'test-cl-{ident}'
    else:
        names=f'cl-{ident}'
        # pre-run
        print('conv_test: prerun')
        model_obj.compile(optimizer=Adam(lr=PRERUN_LR))
        model_obj.fit_gen(
                pdata=data,
                epochs=PRERUN_EPOCHS,
                train_sz=PRERUN_TRAIN_SIZE,
                valid_sz=PRERUN_VALID_SIZE,
                batch_size=PRERUN_BATCH_SIZE,
                history_name=f'pre-{names}',
                checkpoint_name=f'pre-{names}')
    # full_run
    print('conv_test: run')
    model_obj.compile(optimizer=Adam(lr=INITIAL_LR))
    model_obj.fit_gen(
            pdata=data,
            epochs=epochs,
            train_sz=train_size,
            valid_sz=valid_size,
            batch_size=batch_size,
            history_name=names,
            checkpoint_name=names)


#
# INTERNAL
#
def _ident(conv_layers):
    clstr=str(conv_layers)
    clstr=clstr.replace(' ','').replace('),(','-')
    clstr=clstr.replace(',[','_').replace(',','.')
    return re.sub(r'[(\)\[\]]','',clstr)


def truthy(value):
    """ Stringy Truthyness
    """
    value=str(value).lower().strip(' ')
    return value not in ['none','false','0','nope','','[]']


#
# MAIN
#
def _run(args):
    conv_layers=eval(args.conv_layers)
    batch_norm=truthy(args.batch_norm)
    test_run=truthy(args.test)
    create=truthy(args.create)
    if test_run:
        print('conv_test:TEST_RUN')
        tsize=TEST_TRAIN_SIZE
        vsize=TEST_VALID_SIZE
        bsize=TEST_BATCH_SIZE
        epochs=TEST_EPOCHS
        data=PlanetData(create=create,train_size=TEST_PD_SIZE)
    else:
        bsize=int(args.batch_size)
        epochs=int(args.epochs)
        data=PlanetData(create=create,train_size='FULL')
        train_size=bsize*epochs
        valid_size=math.floor(train_size*VALID_PCT)
    if truthy(args.dry):
        print('\nconv_test:DRY_RUN')
        print('\tconv_layers:',conv_layers)
        print('\tepochs:',epochs)
        print('\tbatch_size:',bsize)
        print('\tbatch_norm:',batch_norm)
        print('\tweights:',args.weights)
        print('\ttest_run:',test_run)
        print('\t=>train_size:',train_size)
        print('\t=>valid_size:',valid_size)
        print('')
    else:
        run(data=data,
            conv_layers=conv_layers,
            batch_size=bsize,
            epochs=epochs,
            train_size=train_size,
            valid_size=valid_size,
            batch_norm=batch_norm,
            weights=args.weights,
            test_run=test_run)



def main():
    parser=argparse.ArgumentParser(description='RUN CONV LAYER TESTS')
    subparsers=parser.add_subparsers()
    """ run """
    parser_run=subparsers.add_parser(
        'run',
        help='run aflex based on non default conv_layers')
    parser_run.add_argument(
        'conv_layers',help='conv_layers string (with quotes). example: \'[(32,[3]),(64,[3]),(16,[3])]\'')
    parser_run.add_argument(
        '-b','--batch_size',
        default=DEFAULT_BATCH_SIZE,
        help=f'batch_size. default {DEFAULT_BATCH_SIZE})')
    parser_run.add_argument(
        '-e','--epochs',
        default=DEFAULT_EPOCHS,
        help=f'number of epochs. default {DEFAULT_EPOCHS}')
    parser_run.add_argument(
        '-w','--weights',
        default=None,
        help=f'path to initial weights')
    parser_run.add_argument(
        '-n','--batch_norm',
        default=DEFAULT_BATCH_NORM,
        help=f'batch normalize after blocks: True | False. default {DEFAULT_BATCH_NORM}')
    parser_run.add_argument(
        '-c','--create',
        default=False,
        help=f'create new dataset: True | False. default False')
    parser_run.add_argument(
        '-t','--test',
        default=DEFUALT_TEST_MODE,
        help=f'is test run: True | False. default {DEFUALT_TEST_MODE}')
    parser_run.add_argument(
        '-d','--dry',
        default=False,
        help=f'is dry run (only print params): True | False. default {DEFUALT_TEST_MODE}')
    parser_run.set_defaults(func=_run)
    """ init """
    args=parser.parse_args()
    args.func(args)


if __name__ == "__main__": 
    main()
