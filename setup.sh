cd /tmp
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh

source ~/.bashrc

cd ~/models/research/audioset
conda env create -f conda.yml
conda activate audiodream

jupyter notebook --generate-config
jupyter notebook password

cd ~
mkdir ssl
cd ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout mykey.key -out mycert.pem

cd ~/models/research/audioset

gdown https://drive.google.com/uc?id=0B49XSFgf-0yVQk01eG92RHg4WTA

sudo apt-get install unzip
unzip packed_features.zip
rm packed_features.zip

jupyter notebook --certfile=~/ssl/mycert.pem --keyfile ~/ssl/mykey.key --no-browser
