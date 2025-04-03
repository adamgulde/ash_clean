# download datasets
DATASETS="datasets"
MODELS="models"
mkdir -p ${DATASETS}

wget -P ${DATASETS} http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz

# download small scale ood datasets
wget -P ${DATASETS} https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
wget -P ${DATASETS} https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz
wget -P ${DATASETS} https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz
wget -P ${DATASETS} https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz

# unpack
tar -xf ${DATASETS}/cifar-10-python.tar.gz -C ${DATASETS}
tar -xf ${DATASETS}/LSUN.tar.gz -C ${DATASETS}
tar -xf ${DATASETS}/LSUN_resize.tar.gz -C ${DATASETS}
tar -xf ${DATASETS}/iSUN.tar.gz -C ${DATASETS}
tar -xf ${DATASETS}/SUN.tar.gz -C ${DATASETS}
