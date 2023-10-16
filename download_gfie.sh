root_url="https://airobot.nankai.edu.cn/"

mkdir GFIE_dataset
cd GFIE_dataset

#Download the rgb file of GFIE

for i in $(seq 1 32)
do
    file_id=`echo $i | awk '{printf("%03d\n",$0)}'`
    wget -c -t 10 $root_url"rgb"$file_id".zip";
done


for i in $(seq 1 32)
do
    file_id=`echo $i | awk '{printf("%03d\n",$0)}'`
    org_file_name="rgb"$file_id".zip"
    replace_file_name="rgb.zip."$file_id
    echo "replace the zip file name: "$org_file_name" "$replace_file_name
    mv $org_file_name $replace_file_name
done

7z x rgb.zip.001

rm -rf rgb.zip.*

# Download the depth file of GFIE

for i in $(seq 1 35)
do
    file_id=`echo $i | awk '{printf("%03d\n",$0)}'`
    wget -c -t 10 $root_url"depth"$file_id".zip";
done

for i in $(seq 1 35)
do
    file_id=`echo $i | awk '{printf("%03d\n",$0)}'`
    org_file_name="depth"$file_id".zip"
    replace_file_name="depth.zip."$file_id
    echo "replace the zip file name: "$org_file_name" "$replace_file_name
    mv $org_file_name $replace_file_name
done

7z x depth.zip.001

rm -rf depth.zip.*

7z x depth.zip

# download the aux file
wget -c -t 10 https://airobot.nankai.edu.cn/annotation_CameraKinect.zip

7z x annotation_CameraKinect.zip

