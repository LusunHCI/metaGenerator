wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

TASK=metarevoew
for SPLIT in train val
do
  for LANG in source target
  do
    python fairseq/examples/roberta/multiprocessing_bpe_encoder.py \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "/content/metareview/$SPLIT.$LANG" \
    --outputs "/content/data/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done


fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "/content/data/train.bpe" \
  --validpref "/content/data/val.bpe" \
  --destdir "/content/data/metareview-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt