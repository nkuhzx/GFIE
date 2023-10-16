mkdir CAD120_dataset
cd CAD120_dataset

# download the rgb file
wget -c -t 10 https://airobot.nankai.edu.cn/rgb.zip

7z x rgb.zip

# download the depth file
wget -c -t 10 https://airobot.nankai.edu.cn/depth.zip

7z x depth.zip

# download the aux file
wget -c -t 10 https://airobot.nankai.edu.cn/cad120_annotation_CameraKinect.zip

7z x cad120_annotation_CameraKinect.zip

