import re
from nltk import ngrams
from peerreview.processing.utils import _get_word_ngrams
def exhaustive_beam_selection(reviews_sents, meta_review, summary_size, k = 5):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    # max_rouge = 0.0
    meta_review = _rouge_clean(meta_review).split()
    sents = [_rouge_clean(s).split() for s in reviews_sents]
    evaluated_1grams = [set(ngrams(sent, 1)) for sent in sents]
    reference_1grams = set(ngrams(meta_review, 1))
    evaluated_2grams = [set(ngrams(sent, 2)) for sent in sents]
    reference_2grams = set(ngrams(meta_review, 2))
    def get_rouge_score(working_set):
        candidates_1 = [evaluated_1grams[idx] for idx in working_set]
        candidates_1 = set.union(*map(set, candidates_1))
        candidates_2 = [evaluated_2grams[idx] for idx in working_set]
        candidates_2 = set.union(*map(set, candidates_2))
        rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
        rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
        return rouge_1 + rouge_2
    existing_working_sets = {}
    def beam_search(working_set, k = k, score=0.0):
        set_hash = ""
        for w in sorted(working_set):
            set_hash += str(w)
        if set_hash in existing_working_sets: return existing_working_sets[set_hash]
        if len(working_set) >= summary_size or len(working_set) >= len(sents):
            existing_working_sets[set_hash] = [(working_set, score)]
            return [(working_set, score)]
        scores = []
        for i in range(len(sents)):
            if (i in working_set): continue
            c = working_set.copy()
            c.add(i)
            rouge_score = get_rouge_score(c)
            scores.append((rouge_score, i))
        scores = sorted(scores, key=lambda x: x[0], reverse=True)
        results = []
        for score, idx in scores[:k]:
            new_working_set = working_set.copy()
            new_working_set.add(idx)
            subtree_res = beam_search(new_working_set, k=k, score=score)
            results += subtree_res
        return results
    res = beam_search(set(), k=k)
    return sorted(res, key = lambda x: x[1], reverse=True)
def beamsearch_selection(reviews_sents, meta_review, summary_size, k = 5):
    """
    returns the top k found sets of sentences from reviews_sents in terms of rouge score when compared to the meta_review text
    
    reviews_sents should be a list of sentences
    
    meta_review should be a single string containing the meta_review
    """
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)
    meta_review = _rouge_clean(meta_review).split()
    sents = [_rouge_clean(s).split() for s in reviews_sents]
    evaluated_1grams = [set(ngrams(sent, 1)) for sent in sents]
    reference_1grams = set(ngrams(meta_review, 1))
    evaluated_2grams = [set(ngrams(sent, 2)) for sent in sents]
    reference_2grams = set(ngrams(meta_review, 2))
    def get_rouge_score(working_set):
        candidates_1 = [evaluated_1grams[idx] for idx in working_set]
        candidates_1 = set.union(*map(set, candidates_1))
        candidates_2 = [evaluated_2grams[idx] for idx in working_set]
        candidates_2 = set.union(*map(set, candidates_2))
        rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
        rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
        return rouge_1 + rouge_2
    existing_working_sets = {}
    
    candidate_working_sets = []
    for _ in range(k):
        candidate_working_sets.append(
            (0.0, set())
        )
    max_rouge = 0.0
    best_set = None
    for s in range(summary_size):
        scores = []
        for _, working_set in candidate_working_sets:
            # for each candidate working set, add new sentence and measure rouge score, of all 
            # len(sents) * len(candidate_working_sets) produced options, keep the best k
            for i in range(len(sents)):
                if (i in working_set): continue
                c = working_set.copy()
                c.add(i)
                hashable_set = frozenset(c)
                if hashable_set in existing_working_sets:
                    continue
                rouge_score = get_rouge_score(c)
                if rouge_score > max_rouge:
                    max_rouge = rouge_score
                    best_set = (max_rouge, c)
                existing_working_sets[hashable_set] = rouge_score
                scores.append((rouge_score, i, c))
        scores = sorted(scores, key=lambda x: x[0], reverse=True)
        candidate_working_sets = []
        for score, _, working_set in scores[:k]:
            candidate_working_sets.append((score, working_set))
    return best_set, candidate_working_sets
def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_selection(reviews_sents, meta_review, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    meta_review = _rouge_clean(meta_review).split()
    sents = [_rouge_clean(s).split() for s in reviews_sents]
    evaluated_1grams = [set(ngrams(sent, 1)) for sent in sents]
    reference_1grams = set(ngrams(meta_review, 1))
    evaluated_2grams = [set(ngrams(sent, 2)) for sent in sents]
    reference_2grams = set(ngrams(meta_review, 2))

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            # print("rouge score", rouge_score)
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected, cur_max_rouge
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected), max_rouge