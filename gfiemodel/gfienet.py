import torch
import torch.nn as nn
from gfiemodel.gfiemodule import PSFoVModule,EGDModule,GGHModule

class GFIENet(nn.Module):

    def __init__(self,pretrained=False):

        super(GFIENet,self).__init__()

        self.psfov_module=PSFoVModule()
        self.egd_module=EGDModule()
        self.ggh_module=GGHModule(pretrained)


    def forward(self,simg,himg,headloc,matrix_T):
        """
        Args:
              simg: scene image
              himg: cropped head image
              headloc: mask representing the position of the head in scene image
              matrix_T: matrix_T: unprojected coordiantes represent by matrix T
        Returns:
              pred_heatmap: A heatmap representing a 2D gaze target
        """


        # Estimate gaze direction
        pred_gazevector=self.egd_module(himg)

        # Perceive Stero FoV
        Stereo_Fov=self.psfov_module(matrix_T,pred_gazevector,simg.shape)

        # Generate gaze heatmap
        pred_heatmap=self.ggh_module(simg,Stereo_Fov,headloc)

        return {"pred_heatmap":pred_heatmap,
                "pred_gazevector":pred_gazevector,
                "Stereo_Fov":Stereo_Fov}

if __name__ == '__main__':

    rgbimg=torch.randn((4,3,224,224))

    faceimg=torch.randn((4,3,224,224))

    head_location=torch.randn((4,1,224,224))

    gaze_vs=torch.randn((4,224,224,3))

    print(rgbimg.shape)

    testnet=GFIENet()

    output=testnet(rgbimg,faceimg,head_location,gaze_vs)

    print(output["pred_gazevector"].shape)

