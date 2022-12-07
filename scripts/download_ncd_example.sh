echo Creating the dataset path...

mkdir -p data
cd data

echo Downloading Newer College dataset, Quad example subset...
wget -O ncd_example.tar.gz -c https://uni-bonn.sciebo.de/s/ZKTMubNY9mqbfwN/download

echo Extracting dataset...
tar -xvf ncd_example.tar.gz

rm ncd_example.tar.gz

cd ../..