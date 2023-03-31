import torch
import torch.backends.cudnn as cudnn
import argparse
import os
import shutil
import sys
import random
import time
import numpy as np
from datetime import datetime

from config import cfg
from dataset.gfie import GFIELoader
from utils.model_utils import init_model,setup_model,save_checkpoint,resume_checkpoint,init_checkpoint

from trainer import Trainer
from tester import Tester
from tensorboardX import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train_engine(opt):

    best_dist_error=sys.maxsize
    best_cosine_error=sys.maxsize
    # init gaze model
    gazemodel=init_model(opt)

    # set criterion and optimizer for gaze model
    criterion,optimizer=setup_model(gazemodel,opt)

    writer=False
    # create log dir for tensorboardx
    if writer is not None:
        opt.OTHER.logdir=os.path.join(opt.OTHER.logdir,
                              datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        if os.path.exists(opt.OTHER.logdir):
            shutil.rmtree(opt.OTHER.logdir)
        os.makedirs(opt.OTHER.logdir)
        writer = SummaryWriter(opt.OTHER.logdir)



    # set random seed for reduce the randomness
    random.seed(opt.OTHER.seed)
    np.random.seed(opt.OTHER.seed)
    torch.manual_seed(opt.OTHER.seed)

    # reduce the randomness
    cudnn.benchmark = False
    cudnn.deterministic=True

    # resume the training or initmodel
    if opt.TRAIN.resume==True:

        if os.path.isfile(opt.TRAIN.resume_add):
            gazemodel, optimizer, best_dist_error,best_cosine_error, opt = resume_checkpoint(gazemodel, optimizer, opt)

        else:
            raise Exception("No such checkpoint file")


    dataloader=GFIELoader(opt)
    train_loader=dataloader.train_loader
    val_loader=dataloader.val_loader
    test_loader=dataloader.test_loader

    # init trainer and validator for gazemodel
    trainer=Trainer(gazemodel,criterion,optimizer,train_loader,val_loader,opt,writer=writer)

    tester=Tester(gazemodel,criterion,test_loader,opt,writer=writer)

    trainer.get_best_error(best_dist_error,best_cosine_error)

    optimizer.zero_grad()


    print("Total epoch:{}".format(opt.TRAIN.end_epoch))


    for epoch in range(opt.TRAIN.start_epoch,opt.TRAIN.end_epoch):


        print("Epoch number:{} | Learning rate:{}\n".format(epoch,optimizer.param_groups[0]["lr"]))

        trainer.train(epoch, opt)

        # save the parameters of model
        if epoch%opt.TRAIN.save_intervel==0:

            valid_error=[trainer.eval_dist.avg,trainer.eval_cosine.avg]

            save_checkpoint(gazemodel,optimizer,valid_error,False,epoch,opt)

        # save the parameters of model with the best performance on valid dataset
        if trainer.best_flag:
            valid_error = [trainer.eval_dist.avg, trainer.eval_cosine.avg]

            save_checkpoint(gazemodel, optimizer, valid_error,trainer.best_flag, epoch, opt)


        time.sleep(0.03)

        dist_error,gaze_error=tester.test(epoch,opt)
        print("current error| L2 dist: {:.2f}/Gaze cosine {:.2f}".format(dist_error,gaze_error))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="GFIE benchmark Model"
    )

    parser.add_argument(
        "--cfg",
        default="config/gfiebenchmark.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=True,
        help="choose if use gpus"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.OTHER.device='cuda:0' if (torch.cuda.is_available() and args.gpu) else 'cpu'
    print("The model running on {}".format(cfg.OTHER.device))

    train_engine(cfg)