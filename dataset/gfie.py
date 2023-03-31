import numpy as np
import pandas as pd
import os

from PIL import Image

import torch

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional  as TF

from utils import img_utils
import matplotlib.pyplot as plt

class GFIELoader(object):

    def __init__(self,opt):

        self.train_gaze = GFIEDataset( 'train', opt, show=False)
        self.val_gaze = GFIEDataset( 'valid', opt, show=False)
        self.test_gaze=GFIEDataset( 'test', opt, show=False)


        self.train_loader=DataLoader(self.train_gaze,
                                     batch_size=opt.DATASET.train_batch_size,
                                     num_workers=opt.DATASET.load_workers,
                                     shuffle=True,
                                     collate_fn=collate_fn)


        self.val_loader=DataLoader(self.val_gaze,
                                   batch_size=opt.DATASET.test_batch_size,
                                   num_workers=opt.DATASET.load_workers,
                                   shuffle=False,
                                   collate_fn=collate_fn)

        self.test_loader=DataLoader(self.test_gaze,
                                   batch_size=opt.DATASET.test_batch_size,
                                   num_workers=opt.DATASET.load_workers,
                                   shuffle=False,
                                   collate_fn=collate_fn)


class GFIEDataset(Dataset):

    def __init__(self,dstype,opt,show=False):


        rgb_path=os.path.join(opt.DATASET.root_dir,opt.DATASET.rgb)
        depth_path=os.path.join(opt.DATASET.root_dir,opt.DATASET.depth)

        camerapara=np.load(os.path.join(opt.DATASET.root_dir,opt.DATASET.camerapara))

        if dstype=="train":
            annofile_path=os.path.join(opt.DATASET.root_dir,opt.DATASET.train)
        elif dstype=="valid":
            annofile_path = os.path.join(opt.DATASET.root_dir, opt.DATASET.valid)
        elif dstype=="test":
            annofile_path = os.path.join(opt.DATASET.root_dir, opt.DATASET.test)
        else:
            raise NotImplemented

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
        transform_list.append(transforms.Resize((self.input_size, self.input_size)))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        self.transform = transforms.Compose(transform_list)

        self.dstype=dstype

        self.imshow=show


    def __getitem__(self, index):

        scene_id,frame_index,h_x_min,h_y_min,h_x_max,h_y_max,eye_u,eye_v,eye_X,eye_Y,eye_Z=self.X_train.iloc[index]
        scene_id=str(int(scene_id))
        frame_index=int(frame_index)

        gaze_u, gaze_v,gaze_X, gaze_Y, gaze_Z = self.Y_train.iloc[index]

        rgb_path=os.path.join(self.rgb_path,self.dstype,"scene{}".format(scene_id),"{:04}.jpg".format(frame_index))
        depth_path=os.path.join(self.depth_path,self.dstype,"scene{}".format(scene_id),"{:04}.npy".format(frame_index))

        # load the rgb image
        img = Image.open(rgb_path)
        img = img.convert('RGB')
        width, height = img.size
        org_width, org_height = width, height

        # load the depth image
        depthimg=np.load(depth_path)
        # replace the invalid value with 0
        depthimg[np.isnan(depthimg)]=0
        depthimg=depthimg.astype(np.float32)
        depthimg=Image.fromarray(depthimg)

        # expand face bbox a bit
        k=0.1
        h_x_min -= k * abs(h_x_max - h_x_min)
        h_y_min -= k * abs(h_y_max - h_y_min)
        h_x_max += k * abs(h_x_max - h_x_min)
        h_y_max += k * abs(h_y_max - h_y_min)

        x_min, y_min, x_max, y_max=map(float,[h_x_min,h_y_min,h_x_max,h_y_max])

        # Data augmentation for training procedure
        offset_x, offset_y = 0, 0
        flip_flag = False
        if self.dstype=="train":

            # Jitter (expansion-only) bounding box size
            if np.random.random_sample() <= 0.5:
                k = np.random.random_sample() * 0.2
                x_min -= k * abs(x_max - x_min)
                y_min -= k * abs(y_max - y_min)
                x_max += k * abs(x_max - x_min)
                y_max += k * abs(y_max - y_min)

            # Random crop
            if np.random.random_sample() <= 0.5:
                # calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
                crop_x_min = np.min([gaze_u , x_min, x_max])
                crop_y_min = np.min([gaze_v , y_min, y_max])
                crop_x_max = np.max([gaze_u , x_min, x_max])
                crop_y_max = np.max([gaze_v , y_min, y_max])

                # randomly select a top left corner
                if crop_x_min >= 0:
                    crop_x_min = np.random.uniform(0, crop_x_min)
                if crop_y_min >= 0:
                    crop_y_min = np.random.uniform(0, crop_y_min)

                # find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
                crop_width_min = crop_x_max - crop_x_min
                crop_height_min = crop_y_max - crop_y_min
                crop_width_max = width - crop_x_min
                crop_height_max = height - crop_y_min
                # randomly select a width and a height
                crop_width = np.random.uniform(crop_width_min, crop_width_max)
                crop_height = np.random.uniform(crop_height_min, crop_height_max)

                # crop scene img
                img = TF.crop(img, crop_y_min, crop_x_min, crop_height, crop_width)

                # crop depth img
                depthimg=TF.crop(depthimg, crop_y_min, crop_x_min, crop_height, crop_width)

                # record the crop's (x, y) offset
                offset_x, offset_y = crop_x_min, crop_y_min

                # convert coordinates into the cropped frame
                x_min, y_min, x_max, y_max = x_min - offset_x, y_min - offset_y, x_max - offset_x, y_max - offset_y

                gaze_u, gaze_v = (gaze_u - offset_x) / float(crop_width), \
                                 (gaze_v - offset_y) / float(crop_height)

                eye_u, eye_v = (eye_u - offset_x) / float(crop_width), \
                                 (eye_v  - offset_y) / float(crop_height)

                width, height = crop_width, crop_height

            else:
                gaze_u, gaze_v = (gaze_u - offset_x) / float(width), \
                                 (gaze_v - offset_y) / float(height)

                eye_u, eye_v = (eye_u - offset_x) / float(width), \
                                 (eye_v  - offset_y) / float(height)

            # Random flip
            if np.random.random_sample() <= 0.5:
                flip_flag=True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                depthimg = depthimg.transpose(Image.FLIP_LEFT_RIGHT)
                x_max_2 = width - x_min
                x_min_2 = width - x_max
                x_max = x_max_2
                x_min = x_min_2
                gaze_u= 1 - gaze_u
                eye_u=1 -eye_u

            # Random change the brightness, contrast and saturation of the scene images
            if np.random.random_sample() <= 0.5:
                img = TF.adjust_brightness(img, brightness_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_contrast(img, contrast_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_saturation(img, saturation_factor=np.random.uniform(0, 1.5))

        else:

            gaze_u, gaze_v = gaze_u  / float(width), \
                             gaze_v  / float(height)

            eye_u, eye_v = eye_u  / float(width), \
                           eye_v  / float(height)

        # represent the head location with mask
        head_channel = img_utils.get_head_box_channel(x_min, y_min, x_max, y_max, width, height,
                                                    resolution=self.input_size, coordconv=False).unsqueeze(0)

        # the final image size in train/valid/test
        final_width,final_height=img.size

        # crop the face
        headimg = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # set for display
        if self.imshow:
            img_show=img
            depthimg_show=depthimg

        # resize scene/face image and convert them to tensor
        if self.transform is not None:

            img=self.transform(img)
            headimg=self.transform(headimg)

        # Generate the matrix_T
        depthmap=depthimg.resize((self.input_size,self.input_size),Image.BICUBIC)
        depthmap=np.array(depthmap)

        # scale proportionally
        scale_width,scale_height=final_width/self.input_size,final_height/self.input_size

        # construct empty matrix
        matrix_T_DW = np.linspace(0, self.input_size - 1, self.input_size)
        matrix_T_DH = np.linspace(0, self.input_size - 1, self.input_size)
        [matrix_T_xx, matrix_T_yy] = np.meshgrid(matrix_T_DW, matrix_T_DH)

        # construct matrix_T according to Eq 3. in paper
        fx,fy,cx,cy=self.camerapara
        if flip_flag:
            cx= org_width - cx
            matrix_T_X = (matrix_T_xx * scale_width + (org_width - final_width - offset_x) - cx) * depthmap / fx

        else:
            matrix_T_X = (matrix_T_xx * scale_width + offset_x - cx) * depthmap / fx

        matrix_T_Y = (matrix_T_yy * scale_height + offset_y - cy) * depthmap / fy
        matrix_T_Z = depthmap

        matrix_T = np.dstack((matrix_T_X, matrix_T_Y, matrix_T_Z))
        matrix_T = matrix_T.reshape([-1, 3])
        matrix_T = matrix_T.reshape([self.input_size, self.input_size, 3])

        if flip_flag:
            matrix_T = matrix_T - np.array([-eye_X, eye_Y, eye_Z])
        else:
            matrix_T = matrix_T - np.array([eye_X, eye_Y, eye_Z])

        norm_value = np.linalg.norm(matrix_T, axis=2, keepdims=True)
        norm_value[norm_value <= 0] = 1

        matrix_T = matrix_T / norm_value

        # convert it to tensor
        matrix_T=torch.from_numpy(matrix_T).float()


        # generate the gaze vector label
        gaze_vector = np.array([gaze_X - eye_X, gaze_Y - eye_Y, gaze_Z - eye_Z])

        if flip_flag:
            gaze_vector[0]=-gaze_vector[0]

        norm_gaze_vector = 1.0 if np.linalg.norm (gaze_vector) <= 0.0 else np.linalg.norm (gaze_vector)
        gaze_vector=gaze_vector/norm_gaze_vector
        gaze_vector=torch.from_numpy(gaze_vector)

        # generate the heat map label
        gaze_heatmap = torch.zeros(self.output_size, self.output_size)  # set the size of the output

        gaze_heatmap = img_utils.draw_labelmap(gaze_heatmap, [gaze_u * self.output_size, gaze_v * self.output_size],
                                               3,type='Gaussian')


        # auxilary info
        gaze_target2d=torch.from_numpy(np.array([gaze_u,gaze_v]))
        matrix_T_heatmap = np.dot(matrix_T, gaze_vector)

        # display
        if self.imshow:

            def unnorm(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
                std = np.array(std).reshape(3, 1, 1)
                mean = np.array(mean).reshape(3, 1, 1)
                return img * std + mean

            figure,ax=plt.subplots(2,3)
            figure.set_size_inches(15 ,8)

            simgshow=unnorm(img.numpy()) * 255
            simgshow=np.clip(simgshow,0,255)
            simgshow=simgshow.astype(np.uint8)

            himgshow=unnorm(headimg.numpy()) * 255
            himgshow=np.clip(himgshow,0,255)
            himgshow=himgshow.astype(np.uint8)

            eyes_outpix=[eye_u*self.input_size,eye_v*self.input_size]

            gaze_outpix = [gaze_u * self.input_size, gaze_v * self.input_size]

            # display scene image
            ax[0][0].imshow(np.transpose(simgshow, (1, 2, 0)))
            # display gaze target and eyes in scene image
            ax[0][0].scatter(eyes_outpix[0],eyes_outpix[1])
            ax[0][0].scatter(gaze_outpix[0],gaze_outpix[1])

            # display depth map
            ax[0][1].imshow(depthmap,cmap='gray')
            # display head image
            ax[1][0].imshow(np.transpose(himgshow, (1, 2, 0)))
            # display expected stero FoV heatmap
            ax[1][1].imshow(matrix_T_heatmap, cmap='jet')


            plt.show()

        all_data={}
        all_data['simg'] = img
        all_data["himg"] = headimg
        all_data["headloc"] = head_channel
        all_data["matrixT"]=matrix_T

        # Y_label
        all_data["gaze_heatmap"] = gaze_heatmap
        all_data["gaze_vector"] = gaze_vector
        all_data["gaze_target2d"] = gaze_target2d

        return all_data


    def __len__(self):

        return self.length

def collate_fn(batch):

    batch_data={}

    batch_data["simg"]=[]
    batch_data["himg"]=[]
    batch_data["headloc"]=[]
    batch_data["matrixT"]=[]


    batch_data["gaze_heatmap"]=[]
    batch_data["gaze_vector"]=[]
    batch_data["gaze_target2d"]=[]


    for data in batch:
        batch_data["simg"].append(data["simg"])
        batch_data["himg"].append(data["himg"])
        batch_data["headloc"].append(data["headloc"])
        batch_data["matrixT"].append(data["matrixT"])

        batch_data["gaze_heatmap"].append(data["gaze_heatmap"])
        batch_data["gaze_vector"].append(data["gaze_vector"])
        batch_data["gaze_target2d"].append(data["gaze_target2d"])


    # train data
    batch_data["simg"]=torch.stack(batch_data["simg"],0)
    batch_data["himg"]=torch.stack(batch_data["himg"],0)
    batch_data["headloc"]=torch.stack(batch_data["headloc"],0)
    batch_data["matrixT"]=torch.stack(batch_data["matrixT"],0)


    # label data
    batch_data["gaze_heatmap"]=torch.stack(batch_data["gaze_heatmap"],0)
    batch_data["gaze_vector"] = torch.stack(batch_data["gaze_vector"], 0)
    batch_data["gaze_target2d"] = torch.stack(batch_data["gaze_target2d"], 0)

    return batch_data
