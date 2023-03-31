from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Default Config definition
# -----------------------------------------------------------------------------

_C=CN()

# -----------------------------------------------------------------------------
# Dataset Path
# -----------------------------------------------------------------------------

_C.DATASET=CN()

# dataset dir
_C.DATASET.root_dir = ""
_C.DATASET.rgb =""
_C.DATASET.depth =""
_C.DATASET.camerapara =""

_C.DATASET.train = ""
_C.DATASET.valid=""
_C.DATASET.test=""


# dataset loader
_C.DATASET.load_workers=24
_C.DATASET.train_batch_size=64
_C.DATASET.valid_batch_size=64
_C.DATASET.test_batch_size=64


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL=CN()

# pre-trained parameters
_C.MODEL.backboneptpath=""


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
# model optimizer and criterion type and relative parameters
_C.TRAIN=CN()


_C.TRAIN.criterion="mixed"
_C.TRAIN.optimizer="adam"

_C.TRAIN.lr=3.5e-4
_C.TRAIN.weightDecay=1e-4


_C.TRAIN.start_epoch=0
_C.TRAIN.end_epoch=200

# model save interval and address
_C.TRAIN.store=''
_C.TRAIN.save_intervel=1

# model resume
_C.TRAIN.resume=False
_C.TRAIN.resume_add=''


# input and output resolution
_C.TRAIN.input_size=224
_C.TRAIN.output_size=64


# -----------------------------------------------------------------------------
# Other Default
# -----------------------------------------------------------------------------
_C.OTHER=CN()

_C.OTHER.seed=235
# if gpu is used
_C.OTHER.device='cpu'

_C.OTHER.cpkt=""

# log for tensorboardx
_C.OTHER.logdir='./logs'

_C.OTHER.global_step=0

_C.OTHER.lossrec_every=10

_C.OTHER.evalrec_every=600

