import os
from pprint import pprint
from dfgen import DFGen
from kgraph.functional import RESNET as R
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau
import argparse

#
# CONFIG
# 
PKR=os.environ['PKR']
base_csv_root=f'{PKR}/datacsvs'
train_csv_tmpl='./{}_{}.csv'

DEFAULT_BATCH_SIZE=16
DEFAULT_EPOCHS=20
DEFAULT_TRAIN_STEPS=250
DEFAULT_VALID_STEPS=150
#
# SETUP
#
sgd=SGD(lr=0.1,momentum=0.9,decay=0.0001)
callbacks=[ReduceLROnPlateau(patience=5)]


#
# MODEL
# 
graph={
    'meta': {
        'network_type': 'RESNET'
    },
    'compile': {
        'loss_func': 'binary_crossentropy',
        'metrics': ['accuracy'],
        'optimizer': sgd
    },
    'inputs': {
        'batch_shape':(None,256,256,4),
    },
    'output': { 
        'units': 1,
        'activation': 'sigmoid' 
    }
}

#
# METHODS
#
def create_tag_csv(typ,tag,filename):
  print("resnet-single: creating <{}>".format(filename))
  basefilename='{}/{}.csv'.format(base_csv_root,typ)
  gen=DFGen(csv_file=basefilename,csv_sep=',')
  gen.reduce_columns(tag,others=False)
  gen.save(path=filename)


def get_csv(typ,tag,batch_size):
  filename=train_csv_tmpl.format(typ,tag)
  if not os.path.isfile(filename): create_tag_csv(typ,tag,filename)
  return DFGen(csv_file=filename,csv_sep=',',batch_size=batch_size,tags=[tag])


def run(
    kgres,
    tag,
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    v=1,
    weights=None,
    train_steps=DEFAULT_TRAIN_STEPS,
    valid_steps=DEFAULT_VALID_STEPS,
    callbacks=callbacks):
  train_gen=get_csv('train',tag,batch_size)
  valid_gen=get_csv('valid',tag,batch_size)
  run_name='rn_single-{}-{}'.format(tag,v)
  print('\n\n{}:\n'.format(run_name))
  if weights: kgres.load_weights(weights)
  kgres.fit_gen(
   epochs=epochs,
   train_gen=train_gen,
   train_steps=train_steps,
   validation_gen=valid_gen,
   validation_steps=valid_steps,
   history_name=run_name,
   checkpoint_name=run_name,
   callbacks=callbacks)


#
# CLI
#
def main():
    parser=argparse.ArgumentParser(description='RUN RESNET ON SINGLE TAG')
    """ run """
    parser.add_argument(
        'tag',help='tag name')
    parser.add_argument(
        '-b','--batch_size',
        default=DEFAULT_BATCH_SIZE,
        help=f'batch_size. default {DEFAULT_BATCH_SIZE})')
    parser.add_argument(
        '-e','--epochs',
        default=DEFAULT_EPOCHS,
        help=f'number of epochs. default {DEFAULT_EPOCHS}')
    parser.add_argument(
        '-v','--version',
        default=1,
        help=f'version')
    parser.add_argument(
        '-w','--weights',
        default=None,
        help=f'path to initial weights')
    parser.add_argument(
        '--steps',
        default=DEFAULT_TRAIN_STEPS,
        help=f'train steps: default {DEFAULT_TRAIN_STEPS}')
    parser.add_argument(
        '--valid_steps',
        default=DEFAULT_VALID_STEPS,
        help=f'valid steps: default {DEFAULT_VALID_STEPS}')
    parser.set_defaults(func=_run)
    """ init """
    args=parser.parse_args()
    args.func(args)


def _run(args):
    kgres=R(graph)
    run(
      kgres,
      args.tag,
      epochs=int(args.epochs),
      batch_size=int(args.batch_size),
      v=args.version,
      weights=args.weights,
      train_steps=int(args.steps),
      valid_steps=int(args.valid_steps))



if __name__ == "__main__": 
    main()


