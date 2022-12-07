echo Creating the dataset path...

mkdir -p data
cd data

echo Downloading MaiCity dataset...
wget https://www.ipb.uni-bonn.de/html/projects/mai_city/mai_city.tar.gz

echo Extracting dataset...
tar -xvf mai_city.tar.gz

cd ../..