if [ -e data/max ]; then
    echo "Error! Dataset already exists @ data/max folder, please delete it first."
    echo " Exiting Script..."
    return
fi
mkdir data
cd ./data
curl -L "https://drive.google.com/uc?export=download&id=1NYll3mq-8D2educCErcbYe_iOBwg74Nu" > max.zip
unzip max.zip
rm max.zip
