import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import pandas as pd

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms

from utils import img_utils
from gfiemodel.gfienet import GFIENet

class GFIETestDataset(Dataset):

    def __init__(self,opt):
        super(GFIETestDataset,self).__init__()

        rgb_path=os.path.join(opt.DATASET.root_dir,opt.DATASET.rgb)
        depth_path=os.path.join(opt.DATASET.root_dir,opt.DATASET.depth)

        camerapara=np.load(os.path.join(opt.DATASET.root_dir,opt.DATASET.camerapara))

        annofile_path = os.path.join(opt.DATASET.root_dir, opt.DATASET.test)


        self.input_size=opt.TRAIN.input_size

        df=pd.read_csv(annofile_path)

        self.X_train = df[['scene_id', 'frame_id', "h_x_min","h_y_min","h_x_max","h_y_max",'eye_u','eye_v','eye_X','eye_Y','eye_Z']]

        self.Y_train = df[['gaze_u', 'gaze_v', 'gaze_X', 'gaze_Y', 'gaze_Z']]

        self.length=len(df)

        self.rgb_path=rgb_path
        self.depth_path=depth_path
        self.camerapara=camerapara

        self.input_size=opt.TRAIN.input_size
        self.output_size=opt.TRAIN.output_size

        transform_list = []
        transform_list.append(transforms.Resize((opt.TRAIN.input_size, opt.TRAIN.input_size)))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        scene_id, frame_index, h_x_min, h_y_min, h_x_max, h_y_max, eye_u, eye_v, eye_X, eye_Y, eye_Z = self.X_train.iloc[index]

        scene_id = str(int(scene_id))
        frame_index = int(frame_index)

        gaze_u, gaze_v,gaze_X, gaze_Y, gaze_Z = self.Y_train.iloc[index]

        rgb_path=os.path.join(self.rgb_path,"test","scene{}".format(scene_id),"{:04}.jpg".format(frame_index))
        depth_path=os.path.join(self.depth_path,"test","scene{}".format(scene_id),"{:04}.npy".format(frame_index))

        head_bbox=[h_x_min, h_y_min, h_x_max, h_y_max]

        eye_3d=np.array([eye_X,eye_Y,eye_Z])
        gaze_target_2d=np.array([gaze_u,gaze_v])
        gaze_target_3d=np.array([gaze_X,gaze_Y,gaze_Z])

        # format input
        format_input=self.format_model_input(rgb_path,depth_path,head_bbox,self.camerapara,eye_3d)

        # generate GT
        groundtruth=self.getGroundTruth(eye_3d,gaze_target_2d,gaze_target_3d)

        all_data={}
        all_data['simg'] = format_input["simg"]
        all_data["himg"] = format_input["himg"]
        all_data["headloc"] = format_input["headloc"]
        all_data["matrixT"]=format_input["matrixT"]
        all_data['depthmap'] = format_input["depthmap"]
        all_data["eye3d"]=eye_3d


        all_data["gt_gaze_vector"]=groundtruth["gaze_vector"]
        all_data["gt_gaze_target2d"]=groundtruth["gaze_target2d"]
        all_data["gt_gaze_target3d"]=groundtruth["gaze_target3d"]

        return all_data

    def __len__(self):
        return self.length

    def format_model_input(self,rgb_path,depth_path,head_bbox,campara,eye_coord):


        # load the rgb image and depth map
        rgbimg=Image.open(rgb_path)
        rgbimg = rgbimg.convert('RGB')

        depthimg = np.load(depth_path)  #Image.open(depth_path)
        depthimg[np.isnan(depthimg)]=0
        depthimg=depthimg.astype(np.float32)
        depthimg=Image.fromarray(depthimg)

        width, height = rgbimg.size
        self.img_para=[width,height]

        # expand the head bounding box (in pixel coordinate )
        x_min, y_min, x_max, y_max=map(float,img_utils.expand_head_box(head_bbox
                                                             ,[width,height]))

        # crop the head
        head = rgbimg.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # represent the head location with mask
        head_loc = img_utils.get_head_box_channel(x_min, y_min, x_max, y_max, width, height,
                                                    resolution=self.input_size, coordconv=False).unsqueeze(0)

        # to tensor
        rgbimg=self.transform(rgbimg)
        headimg=self.transform(head)

        # generate the matrix_T
        depthmap=depthimg.resize((self.input_size,self.input_size),Image.BICUBIC)
        depthmap=np.array(depthmap)
        matrix_T=self.getMatrixT(depthmap,campara,eye_coord)

        # reserved for strategy for 3D gaze-following
        depthvalue=depthimg.copy()
        depthvalue=np.array(depthvalue)

        format_input={}
        format_input['simg']=rgbimg
        format_input['himg']=headimg
        format_input['headloc']=head_loc
        format_input['matrixT']=matrix_T
        format_input['depthmap']=depthvalue

        return format_input

    def getGroundTruth(self,eye3d,gt2d,gt3d):

        img_W, img_H = self.img_para

        gaze_vector=gt3d-eye3d
        norm_gaze_vector = 1.0 if np.linalg.norm(gaze_vector) <= 0.0 else np.linalg.norm(gaze_vector)
        gaze_vector=gaze_vector/norm_gaze_vector

        ground_truth={}
        ground_truth["gaze_vector"]=gaze_vector
        ground_truth["gaze_target2d"]=gt2d/np.array([img_W,img_H])
        ground_truth["gaze_target3d"]=gt3d

        return ground_truth

    def getMatrixT(self,dmap,camera_p,eye_3d):

        img_W,img_H=self.img_para

        fx, fy,cx, cy = camera_p

        # construct empty matrix
        matrix_T_DW = np.linspace(0, self.input_size - 1, self.input_size)
        matrix_T_DH = np.linspace(0, self.input_size - 1, self.input_size)
        [matrix_T_xx, matrix_T_yy] = np.meshgrid(matrix_T_DW, matrix_T_DH)

        scale_width, scale_height = img_W / self.input_size, img_H / self.input_size

        matrix_T_X = (matrix_T_xx*scale_width - cx) * dmap /fx
        matrix_T_Y = (matrix_T_yy*scale_height  - cy) * dmap /fy
        matrix_T_Z = dmap

        matrix_T = np.dstack((matrix_T_X, matrix_T_Y, matrix_T_Z))
        matrix_T = matrix_T.reshape([-1, 3])
        matrix_T = matrix_T.reshape([self.input_size, self.input_size, 3])

        matrix_T = matrix_T- eye_3d

        norm_value = np.linalg.norm(matrix_T, axis=2, keepdims=True)
        norm_value[norm_value <= 0] = 1

        matrix_T = matrix_T / norm_value

        matrix_T=torch.from_numpy(matrix_T).float()

        return matrix_T


class CAD120TestDataset(Dataset):

    def __init__(self,opt):

        super(CAD120TestDataset,self).__init__()

        rgb_path=os.path.join(opt.DATASET.root_dir,opt.DATASET.rgb)
        depth_path=os.path.join(opt.DATASET.root_dir,opt.DATASET.depth)

        camerapara=np.load(os.path.join(opt.DATASET.root_dir,opt.DATASET.camerapara))

        annofile_path = os.path.join(opt.DATASET.root_dir, opt.DATASET.test)

        self.input_size=opt.TRAIN.input_size

        df=pd.read_csv(annofile_path)

        self.X_train = df[['dataset_id', 'frame_index', "x_initial","y_initial","w","h", 'eye_x', 'eye_y', 'eye_X', 'eye_Y','eye_Z']]

        self.Y_train = df[['gaze_x', 'gaze_y', 'gaze_X', 'gaze_Y', 'gaze_Z']]

        self.length=len(df)

        self.rgb_path=rgb_path
        self.depth_path=depth_path
        self.camerapara=camerapara

        self.input_size=opt.TRAIN.input_size
        self.output_size=opt.TRAIN.output_size

        transform_list = []
        transform_list.append(transforms.Resize((opt.TRAIN.input_size, opt.TRAIN.input_size)))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        scene_id, frame_index, h_x_min, h_y_min, h_w, h_h, eye_u, eye_v, eye_X, eye_Y, eye_Z = self.X_train.iloc[index]

        frame_index = int(frame_index)

        gaze_u, gaze_v,gaze_X, gaze_Y, gaze_Z = self.Y_train.iloc[index]

        rgb_path=os.path.join(self.rgb_path,"{}".format(scene_id),"RGB_{}.png".format(frame_index))
        depth_path=os.path.join(self.depth_path,"{}".format(scene_id),"Depth_{}.png".format(frame_index))

        head_bbox=[h_x_min, h_y_min, h_w, h_h]

        eye_3d=np.array([eye_X,eye_Y,eye_Z])
        gaze_target_2d=np.array([gaze_u,gaze_v])
        gaze_target_3d=np.array([gaze_X,gaze_Y,gaze_Z])

        # format input
        format_input=self.format_model_input(rgb_path,depth_path,head_bbox,self.camerapara,eye_3d,index)

        # generate GT
        groundtruth=self.getGroundTruth(eye_3d,gaze_target_2d,gaze_target_3d)


        all_data={}
        all_data['simg'] = format_input["simg"]
        all_data["himg"] = format_input["himg"]
        all_data["headloc"] = format_input["headloc"]
        all_data["matrixT"]=format_input["matrixT"]
        all_data['depthmap'] = format_input["depthmap"]
        all_data["eye3d"]=eye_3d

        all_data["gt_gaze_vector"]=groundtruth["gaze_vector"]
        all_data["gt_gaze_target2d"]=groundtruth["gaze_target2d"]
        all_data["gt_gaze_target3d"]=groundtruth["gaze_target3d"]

        return all_data

    def __len__(self):
        return self.length

    def format_model_input(self,rgb_path,depth_path,head_bbox,campara,eye_coord,index=None):

        rgbimg=Image.open(rgb_path)
        rgbimg = rgbimg.convert('RGB')

        depthimg= Image.open(depth_path)
        depthimg= np.array(depthimg).astype(np.float)
        depthimg= depthimg/1000.

        depthimg=Image.fromarray(depthimg)

        width, height = rgbimg.size
        self.img_para=[width,height]

        # convert to image coordinate system
        head_bbox=np.array(head_bbox)
        head_bbox[2:]=head_bbox[2:]+head_bbox[0:2]
        head_bbox=head_bbox*np.array([width,height,width,height])

        # expand the head bounding box (in pixel coordinate )
        x_min, y_min, x_max, y_max=map(float,img_utils.expand_head_box(head_bbox
                                                             ,[width,height]))

        # crop the head
        head = rgbimg.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # represent the head location with mask
        head_loc = img_utils.get_head_box_channel(x_min, y_min, x_max, y_max, width, height,
                                                    resolution=self.input_size, coordconv=False).unsqueeze(0)

        # to tensor
        rgbimg=self.transform(rgbimg)
        headimg=self.transform(head)

        # generate the matrix_T
        depthmap=depthimg.resize((self.input_size,self.input_size),Image.NEAREST)
        depthmap=np.array(depthmap)

        matrix_T=self.getMatrixT(depthmap,campara,eye_coord,index)

        # reserved for strategy for 3D gaze-following
        depthvalue=depthimg.copy()
        depthvalue=np.array(depthvalue)

        format_input={}
        format_input['simg']=rgbimg
        format_input['himg']=headimg
        format_input['headloc']=head_loc
        format_input['matrixT']=matrix_T
        format_input['depthmap']=depthvalue

        return format_input

    def getGroundTruth(self,eye3d,gt2d,gt3d):

        gaze_vector=gt3d-eye3d
        norm_gaze_vector = 1.0 if np.linalg.norm(gaze_vector) <= 0.0 else np.linalg.norm(gaze_vector)
        gaze_vector=gaze_vector/norm_gaze_vector

        ground_truth={}
        ground_truth["gaze_vector"]=gaze_vector
        ground_truth["gaze_target2d"]=gt2d
        ground_truth["gaze_target3d"]=gt3d

        return ground_truth

    def getMatrixT(self,dmap,camera_p,eye_3d,index=None):

        img_W,img_H=self.img_para
        fx, fy,cx, cy = camera_p

        # construct empty matrix
        matrix_T_DW = np.linspace(0, self.input_size - 1, self.input_size)
        matrix_T_DH = np.linspace(0, self.input_size - 1, self.input_size)
        [matrix_T_xx, matrix_T_yy] = np.meshgrid(matrix_T_DW, matrix_T_DH)

        scale_width, scale_height = img_W / self.input_size, img_H / self.input_size

        matrix_T_X = (matrix_T_xx*scale_width - cx) * dmap /fx
        matrix_T_Y = (matrix_T_yy*scale_height  - cy) * dmap /fy
        matrix_T_Z = dmap

        matrix_T = np.dstack((matrix_T_X, matrix_T_Y, matrix_T_Z))

        matrix_T = matrix_T- eye_3d

        norm_value = np.linalg.norm(matrix_T, axis=2, keepdims=True)
        norm_value[norm_value <= 0] = 1

        matrix_T = matrix_T / norm_value
        matrix_T[dmap==0]=np.zeros_like(matrix_T[dmap==0])
        matrix_T=torch.from_numpy(matrix_T).float()

        return matrix_T

def collate_fn(batch):

    batch_data={}

    batch_data["simg"]=[]
    batch_data["himg"]=[]
    batch_data["headloc"]=[]
    batch_data["matrixT"]=[]

    # for inference
    batch_data["depthmap"] = []
    batch_data["eye3d"]=[]

    batch_data["gt_gaze_vector"]=[]
    batch_data["gt_gaze_target2d"]=[]
    batch_data["gt_gaze_target3d"]=[]


    for data in batch:
        batch_data["simg"].append(data["simg"])
        batch_data["himg"].append(data["himg"])
        batch_data["headloc"].append(data["headloc"])
        batch_data["matrixT"].append(data["matrixT"])

        batch_data["depthmap"].append(data["depthmap"])
        batch_data["eye3d"].append(data["eye3d"])

        batch_data["gt_gaze_vector"].append(data["gt_gaze_vector"])
        batch_data["gt_gaze_target2d"].append(data["gt_gaze_target2d"])
        batch_data["gt_gaze_target3d"].append(data["gt_gaze_target3d"])


    # train data
    batch_data["simg"]=torch.stack(batch_data["simg"],0)
    batch_data["himg"]=torch.stack(batch_data["himg"],0)
    batch_data["headloc"]=torch.stack(batch_data["headloc"],0)
    batch_data["matrixT"]=torch.stack(batch_data["matrixT"],0)

    # aux data
    batch_data["depthmap"]=np.stack(batch_data["depthmap"],0)
    batch_data["eye3d"]=np.stack(batch_data["eye3d"],0)

    # label data
    batch_data["gt_gaze_vector"]=np.stack(batch_data["gt_gaze_vector"],0)
    batch_data["gt_gaze_target2d"] = np.stack(batch_data["gt_gaze_target2d"], 0)
    batch_data["gt_gaze_target3d"] = np.stack(batch_data["gt_gaze_target3d"], 0)

    return batch_data


def model_init(device,cpkt):
    cudnn.deterministic = True

    model = GFIENet(pretrained=False)

    model = model.to(device)
    model.eval()

    checkpoint = torch.load(cpkt)
    model.load_state_dict(checkpoint["state_dict"])

    return model

def strategy3dGazeFollowing(depthmap,pred_gh,pred_gv,eye_3d,campara,ratio=0.1):

    img_H,img_W=depthmap.shape

    # get the center of 2d proposal area
    output_h, output_w = pred_gh.shape

    pred_center = list(img_utils.argmax_pts(pred_gh))
    pred_gazetarget_2d=np.array([pred_center[0]/output_w,pred_center[1]/output_h])

    pred_center[0] = pred_center[0] * img_W / output_w
    pred_center[1] = pred_center[1] * img_H / output_h

    # get the proposal rectangle area
    pu_min = pred_center[0] - img_W * ratio / 2
    pu_max = pred_center[0] + img_W * ratio / 2

    pv_min = pred_center[1] - img_H * ratio / 2
    pv_max = pred_center[1] + img_H * ratio / 2

    if pu_min < 0:
        pu_min, pu_max = 0, img_W * ratio
    elif pu_max > img_W:
        pu_max, pu_min = img_W, img_W - img_W * ratio

    if pv_min < 0:
        pv_min, pv_max = 0, img_H * ratio
    elif pv_max > img_H:
        pv_max, pv_min = img_H, img_H - img_H * ratio

    pu_min, pu_max, pv_min, pv_max = map(int, [pu_min, pu_max, pv_min, pv_max])

    # unproject to 3d proposal area
    range_depthmap = depthmap[pv_min:pv_max, pu_min:pu_max]
    fx, fy ,cx, cy = campara

    range_space_DW = np.linspace(pu_min, pu_max - 1, pu_max - pu_min)
    range_space_DH = np.linspace(pv_min, pv_max - 1, pv_max - pv_min)
    [range_space_xx, range_space_yy] = np.meshgrid(range_space_DW, range_space_DH)



    range_space_X = (range_space_xx - cx) * range_depthmap / fx
    range_space_Y = (range_space_yy - cy) * range_depthmap / fy
    range_space_Z = range_depthmap

    proposal_3d = np.dstack([range_space_X, range_space_Y, range_space_Z])

    matrix_T = proposal_3d-eye_3d

    norm_value = np.linalg.norm(matrix_T, axis=2, keepdims=True)
    norm_value[norm_value <= 0] = 1
    matrix_T = matrix_T / norm_value

    # filter out the invalid depth
    matrix_T[range_depthmap == 0] = 0

    # find the
    gaze_vector_similar_set = np.dot(matrix_T, pred_gv)

    max_index_u, max_index_v = img_utils.argmax_pts(gaze_vector_similar_set)

    pred_gazetarget_3d=proposal_3d[int(max_index_v),int(max_index_u)]

    pred_gazevector=matrix_T[int(max_index_v),int(max_index_u)]

    pred_gazetarget_3d=np.array(pred_gazetarget_3d).reshape(-1,3)
    pred_gazetarget_2d=np.array(pred_gazetarget_2d).reshape(-1,2)
    pred_gazevector=pred_gazevector.reshape(-1,3)

    return {"pred_gazetarget_3d":pred_gazetarget_3d,
            "pred_gazetarget_2d":pred_gazetarget_2d,
            "pred_gazevector":pred_gazevector}


