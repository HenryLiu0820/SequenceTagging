# conda activate env1
root=/Users/henryliu/Documents/GitHub/SequenceTagging
cd ${root}

name=BiLSTM-CRF
load=False
datadir='/Users/henryliu/Documents/GitHub/data/project3'
ckpt_path=${root}/checkpoints/${name}
vocab_size=100000

mkdir -p ${ckpt_path}
cd src
python train.py \
    --name ${name} \
    --load ${load} \
    --datadir ${datadir} \
    --vocab_size ${vocab_size} \
    --ckpt_path ${ckpt_path} \