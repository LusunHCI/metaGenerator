!python /content/metaGenerator/fairseq/examples/bart/summarize.py \
  --model-dir /content/checkpoints \
  --model-file checkpoint_best.pt \
  --src /content/metaGenerator/preprocess/metareview/test.source \
  --out /content/metaGenerator/preprocess/metareview/test.hypo