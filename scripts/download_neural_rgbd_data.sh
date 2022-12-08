echo Creating the dataset path...

mkdir -p data
cd data

echo Downloading Neural RGBD dataset...
wget http://kaldir.vc.in.tum.de/neural_rgbd/neural_rgbd_data.zip

echo Extracting dataset...
unzip neural_rgbd_data.zip

rm neural_rgbd_data.zip

cd ../..