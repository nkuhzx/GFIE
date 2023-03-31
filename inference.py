import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from utils.utils import AverageMeter,auc
from tqdm import tqdm
from config import cfg
from utils.infer_engine import GFIETestDataset,collate_fn,model_init,strategy3dGazeFollowing
from utils.infer_engine import CAD120TestDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

@ torch.no_grad()
def inference(cfg,mode="gfie"):

    # init the model
    device=cfg.OTHER.device

    gazemodel=model_init(device,cfg.OTHER.cpkt)

    if mode=="gfie":
        test_dataset=GFIETestDataset(cfg)
    elif mode=="cad120":
        test_dataset=CAD120TestDataset(cfg)
    else:
        raise NotImplemented

    test_loader = DataLoader(test_dataset,
                                  batch_size=cfg.DATASET.test_batch_size,
                                  num_workers=cfg.DATASET.load_workers,
                                  shuffle=False,
                                  collate_fn=collate_fn)

    eval_L2dist_counter=AverageMeter()
    eval_3Ddist_counter=AverageMeter()
    eval_AngleError_counter=AverageMeter()
    eval_AUC_counter=AverageMeter()

    pbar=tqdm(total=len(test_loader))
    for i,data in enumerate(test_loader,0):

        x_simg, x_himg, x_hc = data["simg"], data["himg"], data["headloc"]

        x_matrixT = data["matrixT"]

        x_simg = x_simg.to(device)
        x_himg = x_himg.to(device)
        x_hc = x_hc.to(device)
        x_matrixT = x_matrixT.to(device)

        bs = x_simg.size(0)

        i_depmap=data["depthmap"]
        i_eye3d=data["eye3d"]
        i_img_size=[np.array([i_depmap[i].shape[1],i_depmap[i].shape[0]])[np.newaxis,:] for i in range(i_depmap.shape[0])]
        i_img_size=np.concatenate(i_img_size,axis=0)

        # predict gaze heatmap and gaze vector
        outs = gazemodel(x_simg, x_himg, x_hc, x_matrixT)

        pred_heatmap = outs['pred_heatmap']
        pred_heatmap = pred_heatmap.squeeze(1)
        pred_heatmap = pred_heatmap.data.cpu().numpy()


        pred_gazevector = outs['pred_gazevector']
        pred_gazevector = pred_gazevector.data.cpu().numpy()


        gaze_vector = data["gt_gaze_vector"]
        gaze_target2d = data["gt_gaze_target2d"]
        gaze_target3d = data["gt_gaze_target3d"]

        # strategy for 3d gaze-following and evaluation

        pred_gazevector_list=[]
        pred_gazetarget2d_list=[]
        pred_gazetarget3d_list=[]
        for b_idx in range(bs):

            cur_depmap=i_depmap[b_idx]
            cur_pred_gazeheatmap=pred_heatmap[b_idx]
            cur_pred_gazevector=pred_gazevector[b_idx]
            cur_eye_3d=i_eye3d[b_idx]

            pred_result=strategy3dGazeFollowing(cur_depmap,cur_pred_gazeheatmap,cur_pred_gazevector,cur_eye_3d,test_dataset.camerapara)

            pred_gazevector_list.append(pred_result["pred_gazevector"])
            pred_gazetarget2d_list.append(pred_result["pred_gazetarget_2d"])
            pred_gazetarget3d_list.append(pred_result["pred_gazetarget_3d"])


        pred_gazetarget3d =np.concatenate(pred_gazetarget3d_list,axis=0)
        pred_gazetarget2d =np.concatenate(pred_gazetarget2d_list,axis=0)
        pred_gazevector=np.concatenate(pred_gazevector_list,axis=0)

        # evaluation
        eval_batch_3Ddist=np.sum(np.linalg.norm(pred_gazetarget3d-gaze_target3d,axis=1))/bs
        eval_batch_l2dist=np.sum(np.linalg.norm(pred_gazetarget2d-gaze_target2d,axis=1))/bs

        eval_batch_cosine_similarity=np.sum(pred_gazevector*gaze_vector,axis=1)
        eval_batch_angle_error=np.arccos(eval_batch_cosine_similarity)
        eval_batch_angle_error=np.sum(np.rad2deg(eval_batch_angle_error))/bs

        eval_batch_auc=auc(gaze_target2d,pred_heatmap,i_img_size)
        eval_AUC_counter.update(eval_batch_auc,bs)
        eval_L2dist_counter.update(eval_batch_l2dist,bs)
        eval_3Ddist_counter.update(eval_batch_3Ddist,bs)
        eval_AngleError_counter.update(eval_batch_angle_error,bs)


        pbar.set_postfix(eval_3D_dist=eval_3Ddist_counter.avg,
                         eval_L2_dist=eval_L2dist_counter.avg,
                         eval_Angle_error=eval_AngleError_counter.avg,
                         eval_AUC=eval_AUC_counter.avg)

        pbar.update(1)

    pbar.close()




if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="GFIE benchmark Model"
    )

    parser.add_argument(
        "--cfg",
        default="config/default.yaml",
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
        "--mode",
        default="gfie",
        help="choose a dataset to evaluate",
        type=str,
    )

    args=parser.parse_args()

    if args.mode=="cad120":
        args.cfg="config/cad120evaluation.yaml"
    elif args.mode=="gfie":
        args.cfg="config/gfiebenchmark.yaml"
    else:
        raise NotImplementedError("Please select the correct dataset for evalution (gfie or cad120)")

    cfg.merge_from_file(args.cfg)

    cfg.OTHER.device='cuda:0' if (torch.cuda.is_available() and args.gpu) else 'cpu'
    print("The model running on {}".format(cfg.OTHER.device))

    inference(cfg,mode=args.mode)

