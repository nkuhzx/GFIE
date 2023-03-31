import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from sklearn.metrics import average_precision_score
from PIL import Image
from sklearn.metrics import roc_auc_score
import time

class AverageMeter():

    def __init__(self):

        self.reset()

    def reset(self):

        self.count=0
        self.newval=0
        self.sum=0
        self.avg=0

    def update(self,newval,n=1):

        self.newval=newval
        self.sum+=newval*n
        self.count+=n
        self.avg=self.sum/self.count

class MovingAverageMeter():

    def __init__(self,max_len=30):

        self.max_len=max_len

        self.reset()

    def reset(self):

        self.dq=deque(maxlen=self.max_len)
        self.count=0
        self.avg=0
        self.sum=0


    def update(self,newval):

        self.dq.append(newval)
        self.count=len(self.dq)
        self.sum=np.array(self.dq).sum()
        self.avg=self.sum/float(self.count)

def argmax_pts(heatmap):

    idx=np.unravel_index(heatmap.argmax(),heatmap.shape)
    pred_y,pred_x=map(float,idx)

    return pred_x,pred_y


# # Metric functions
def euclid_dist(pred,target):

    batch_dist=0.
    batch_size=pred.shape[0]
    pred_H,pred_W=pred.shape[1:]

    for b_idx in range(batch_size):

        pred_x,pred_y=argmax_pts(pred[b_idx])
        norm_p=np.array([pred_x,pred_y])/np.array([pred_W,pred_H])


        sample_target=target[b_idx]
        sample_target=sample_target.view(-1,2).numpy()

        sample_dist=np.linalg.norm(sample_target-norm_p)

        batch_dist+=sample_dist

    euclid_dist=batch_dist/float(batch_size)

    return euclid_dist

def auc(gt_gaze,pred_heatmap,imsize):
    batch_size=len(gt_gaze)
    auc_score_list=[]
    for b_idx in range(batch_size):

        multi_hot=multi_hot_targets(gt_gaze[b_idx],imsize[b_idx])
        scaled_heatmap=Image.fromarray(pred_heatmap[b_idx]).resize(size=(imsize[b_idx][0],imsize[b_idx][1]),
                                                                   resample=0)

        scaled_heatmap=np.array(scaled_heatmap)
        sample_auc_score=roc_auc_score(np.reshape(multi_hot,multi_hot.size),
                                np.reshape(scaled_heatmap,scaled_heatmap.size))

        auc_score_list.append(sample_auc_score)

    auc_score=sum(auc_score_list)/len(auc_score_list)
    return auc_score



def ap(label,pred):
    return average_precision_score(label,pred)

def multi_hot_targets(gaze_pts,out_res):
    w,h= out_res
    target_map=np.zeros((h,w))
    # for p in gaze_pts:
    if gaze_pts[0]>=0:
        x,y=map(int,[gaze_pts[0]*float(w),gaze_pts[1]*float(h)])
        x=min(x,w-1)
        y=min(y,h-1)
        target_map[y,x]=1
    return target_map

def draw_labelmap(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    # img = to_numpy(img)

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] += g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    img = img/np.max(img) # normalize heatmap so it has max value of 1
    return img

def cosine_sim(pred_gv,target_gv):

    cosine=pred_gv*target_gv
    # print(np.linalg.norm(pred_gv,axis=1),np.linalg.norm(target_gv,axis=1))
    cosine=np.sum(cosine,axis=1)/(np.linalg.norm(pred_gv,axis=1)*np.linalg.norm(target_gv,axis=1))
    cosine[cosine<1e-6]=1e-6
    cosine=np.sum(cosine)/cosine.shape[0]

    # if cosine is np.nan:
    #     print(cosine)
    #     raise NotImplemented

    return cosine


def unnorm(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    std = np.array(std).reshape(1, 1, 3)
    mean = np.array(mean).reshape(1, 1, 3)
    return img * std + mean

def visualized(rgb_image,face_image,pred_heatmap,gt_heatmap,gt_gaze):

    batch_size=pred_heatmap.shape[0]

    output_shape=pred_heatmap.shape[1:]
    H,W=rgb_image.shape[2:]




    gt_mean_dist,bs_dist=euclid_dist(pred_heatmap, gt_gaze, type='retained')


    gt_gaze=gt_gaze.numpy()



    for index in range(batch_size):

        print("index, avg_dist",index,bs_dist[index])

        figure, ax = plt.subplots(2, 4)
        figure.set_size_inches(20, 8)

        # scene image
        cur_img=rgb_image[index]
        cur_img=cur_img.swapaxes(0,1).swapaxes(1,2)
        cur_img = unnorm(cur_img) * 255
        cur_img = np.clip(cur_img, 0, 255)
        cur_img = cur_img.astype(np.uint8)

        # face image
        cur_face=face_image[index]
        cur_face=cur_face.swapaxes(0,1).swapaxes(1,2)
        cur_face=unnorm(cur_face)*255
        cur_face = np.clip(cur_face, 0, 255)
        cur_face = cur_face.astype(np.uint8)

        # ground truth gaze

        cur_gaze=gt_gaze[index]

        cur_gaze=cur_gaze[cur_gaze!=[-1,-1]]


        cur_gaze=cur_gaze.reshape([-1,2])
        cur_gaze[:,1]=cur_gaze[:,1]*H
        cur_gaze[:,0]=cur_gaze[:,0]*W


        ax[0][0].scatter(cur_gaze[:,0],cur_gaze[:,1],s=1,c='white')
        ax[0][0].imshow(cur_img)


        ax[0][1].imshow(cur_face)

        ax[1][0].imshow(pred_heatmap[index],cmap='jet')

        ax[1][1].imshow(gt_heatmap[index],cmap='jet')

        # ax[0][2].imshow(pred_heatmap[index]+xz_heatmap[index]*0.5)


        plt.show()


