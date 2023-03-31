import torch
import torch.nn as nn
import numpy as np
from utils.utils import AverageMeter,MovingAverageMeter,euclid_dist,visualized,cosine_sim
from tqdm import tqdm

class Tester(object):

    def __init__(self,model,criterion,testloader,opt,writer=None):

        self.model=model
        self.criterion=criterion

        self.testloader=testloader

        self.test_dist=AverageMeter()
        self.test_cosine=AverageMeter()

        self.device=torch.device(opt.OTHER.device)

        self.opt=opt
        self.writer=writer

    @torch.no_grad()
    def test(self,epoch,opt):

        self.model.eval()

        self.test_dist.reset()
        self.test_cosine.reset()

        loader_capacity=len(self.testloader)
        pbar=tqdm(total=loader_capacity)

        for i,data in enumerate(self.testloader,0):


            x_simg, x_himg, x_hc = data["simg"], data["himg"], data["headloc"]

            x_matrixT=data["matrixT"]

            gaze_vector=data["gaze_vector"]
            gaze_target2d = data["gaze_target2d"]

            x_simg=x_simg.to(self.device)
            x_himg=x_himg.to(self.device)
            x_hc=x_hc.to(self.device)
            x_matrixT=x_matrixT.to(self.device)

            inputs_size=x_simg.size(0)

            outs=self.model(x_simg, x_himg, x_hc,x_matrixT)


            pred_heatmap=outs['pred_heatmap']
            pred_heatmap=pred_heatmap.squeeze(1)
            pred_heatmap=pred_heatmap.data.cpu().numpy()

            pred_gazevector=outs['pred_gazevector']
            pred_gazevector=pred_gazevector.data.cpu().numpy()


            gaze_vector=gaze_vector.numpy()


            distval = euclid_dist(pred_heatmap, gaze_target2d)
            cosineval=cosine_sim(pred_gazevector,gaze_vector)

            self.test_dist.update(distval,inputs_size)
            self.test_cosine.update(cosineval,inputs_size)

            pbar.set_postfix(dist=self.test_dist.avg,
                             cosine=self.test_cosine.avg)
            pbar.update(1)

        pbar.close()

        if self.writer is not None:

            self.writer.add_scalar("Eval dist", self.test_dist.avg, global_step=opt.OTHER.global_step)
            self.writer.add_scalar("Eval cosine", self.test_cosine.avg, global_step=opt.OTHER.global_step)

        return self.test_dist.avg,self.test_cosine.avg