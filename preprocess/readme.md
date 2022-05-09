# preprocess data files into BPE format
# Add the path to the python folder
import os
os.environ['PYTHONPATH'] += ":/content/fairseq"

# run the script
bash script/preprocess.sh 

!zip -r metareview-bin.zip data/metareview/