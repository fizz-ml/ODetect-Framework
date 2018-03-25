if [ -e data/max ]; then
    echo "Error! Dataset already exists @ data/max folder, please delete it first."
    echo " Exiting Script..."
    return
fi
mkdir data
cd ./data
curl -L "https://drive.google.com/file/d/1vbe0wOwipm-p1tNDcFuiDE0YFa3qwUyV" > max.zip
unzip max.zip
rm max.zip
