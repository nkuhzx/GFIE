import torch
import torch.nn as nn
import torch.optim as optim
import os

from gfiemodel.gfienet import GFIENet

def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv')!=-1 :
        nn.init.kaiming_normal_(m.weight.data)
    elif classname.find('Linear')!=-1:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias,0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.)
        m.bias.data.zero_()

def init_model(opt):


    # encoder in GGH module pretrained on ImageNet
    model=GFIENet(pretrained=True)

    # backbone in EGD module pretrained on Gaze360
    model.egd_module.backbone.load_state_dict(torch.load(opt.MODEL.backboneptpath))

    model.egd_module.fc.apply(weights_init)
    model.psfov_module.apply(weights_init)
    model.ggh_module.decoder.apply(weights_init)

    model=model.to(opt.OTHER.device)
    return model


def setup_model(model,opt):


    if opt.TRAIN.criterion=="mixed":
        criterion=[nn.MSELoss(reduction='none'),nn.CosineSimilarity()]
    else:
        raise NotImplemented


    if opt.TRAIN.optimizer=="adam":

        optimizer=optim.Adam(model.parameters(),
                             lr=opt.TRAIN.lr,
                             weight_decay=opt.TRAIN.weightDecay)
    else:
        raise NotImplemented

    return criterion,optimizer


def save_checkpoint(model,optimizer,best_error,best_flag,epoch,opt):

    cur_state={
        'epoch':epoch+1,
        'state_dict':model.state_dict(),
        'best_dist_err':best_error[0],
        'best_cosine_err':best_error[1],
        'optimizer':optimizer.state_dict()
    }

    epochnum=str(epoch)

    if best_flag:
        filename='gfiemodel'+'_'+'best.pth.tar'
    else:
        filename='gfiemodel'+'_'+epochnum+'epoch.pth.tar'

    torch.save(cur_state,os.path.join(opt.TRAIN.store,filename))


def resume_checkpoint(model,optimizer,opt):

    checkpoint=torch.load(opt.TRAIN.resume_add)
    opt.TRAIN.start_epoch=checkpoint['epoch']
    best_dist_error=checkpoint['best_dist_err']
    best_cosine_error=checkpoint['best_cosine_err']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    print("=> Loading checkpoint '{}' (epoch {})".format(opt.TRAIN.resume,opt.TRAIN.start_epoch))

    return model, optimizer, best_dist_error,best_cosine_error, opt

def init_checkpoint(model,opt):


    checkpoint=torch.load(opt.TRAIN.initmodel_add)

    model.load_state_dict(checkpoint['state_dict'])

    print("=> Loading init checkpoint ".format(opt.TRAIN.initmodel))

    return model