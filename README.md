# metaGenerator



# Fairseq:
- Task: define data loader, loss, 
	- TODO: file tasks/translation_review.py
- Model: define parameters, (word embedding, pos embedding, review id embedding, review score embedding, etc)
	- TODO: file models/tranformer/transformer_encoder.py
- Dataset: define data structure. TODO: change from (src -> tgt sequences) => (src, reviewer-id, reviewer-score -> tgt)
	- TODO: file data/language_pair_dataset.py => data/review_pair_dataset.py