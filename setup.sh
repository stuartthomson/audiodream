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

curl http://storage.googleapis.com/eu_audioset/youtube_corpus/v1/features/features.tar.gz | tar -xvz

jupyter notebook --certfile=~/ssl/mycert.pem --keyfile ~/ssl/mykey.key --no-browser
