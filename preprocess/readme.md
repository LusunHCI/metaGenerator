# preprocess data files into BPE format
# Add the path to the python folder
import os
os.environ['PYTHONPATH'] += ":/content/metaGenerator/fairseq"

# download dicts
!wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
!wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
!wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'
# add special tokens into dict and encode.json. here is "<BRK>" : code changed in the gpt2_bpe_utils.py

# run the script to generate files under metareview-bin
%cd metaGenerator
!bash preprocess.sh

!zip -r metareview-bin.zip data/metareview/