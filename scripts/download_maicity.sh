echo Creating the dataset path...

mkdir -p data
cd data

echo Downloading MaiCity dataset...
wget https://www.ipb.uni-bonn.de/html/projects/mai_city/mai_city.tar.gz

echo Extracting dataset...
tar -xvf mai_city.tar.gz

echo Downloading MaiCity ground truth point cloud generated from sequence 02 and the ground truth model ...
cd mai_city
wget -O gt_map_pc_mai.ply -c https://uni-bonn.sciebo.de/s/DAMWVCC1Kxkfkyz/download
cd ..

rm mai_city.tar.gz

cd ../..