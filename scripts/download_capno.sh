if [ -e data/capno ]; then
    echo "Error! Dataset already already exists @ data/capno folder, please delete it first."
    echo " Exiting Script..."
    return
fi
wget "http://www.capnobase.org/uploads/media/TBME2013-PPGRR-Benchmark_R3.zip"
mkdir data
mv TBME2013-PPGRR-Benchmark_R3.zip ./data
cd ./data
mkdir capno
unzip TBME2013-PPGRR-Benchmark_R3.zip
mv TBME2013-PPGRR-Benchmark_R3/data capno/raw
mv TBME2013-PPGRR-Benchmark_R3/README.txt capno/raw
rm TBME2013-PPGRR-Benchmark_R3.zip
rm TBME2013-PPGRR-Benchmark_R3 -r
cd ../
