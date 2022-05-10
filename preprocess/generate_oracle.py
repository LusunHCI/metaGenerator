import sys
import pandas as pd
import ast
import numpy as np
import nltk
import pickle
import textblob
from processing import dataset_preprocessing
from processing.data_builder import cal_rouge, greedy_selection, beamsearch_selection
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm import tqdm
from typing import Dict, List

def endswith(sent: str, extensions: List[str]):
    for extension in extensions:
        if sent.endswith(extension):
            return True
    return False


def contain_open_bracket(text: str):
    has_open_bracket = False
    for c in text:
        if c == '(':
            has_open_bracket = True
        if has_open_bracket and c == ')':
            has_open_bracket = False
    return has_open_bracket

def get_sents(text: str) -> List[str]:
    """ Give a text string, return the sentence list """
    # Here are some heuristics that we use to get appropriate sentence splitter.
    # 1. combine sentences with its successor when certain conditions satisfied
    sent_list: List[str] = nltk.tokenize.sent_tokenize(text)
    new_sent_list = [sent.replace("\n", "") for sent in sent_list]
    postprocessed = []
    buff = ""
    for sent in new_sent_list:
        if endswith(sent, ['i.e.', 'i.e .', 'e.g.', 'e.g .', 'resp.', 'resp .',
                           'et al.', 'et al .', 'i.i.d.', 'i.i.d .', 'Eq.',
                           'Eq .', 'eq.', 'eq .', 'incl.', 'incl .', 'Fig.',
                           'Fig .', 'w.r.t.', 'w.r.t .', 'sec.', 'sec .',
                           'Sec.', 'Sec .']) or len(sent) < 10 \
                or contain_open_bracket(sent):
            buff += sent
        else:
            postprocessed.append(buff + sent)
            buff = ""
    if len(buff) > 0:
        postprocessed.append(buff)
    return postprocessed

def prepare_inputs(text: str) -> List[List[str]]:
    sents = get_sents(text)
#     words = [nltk.word_tokenize(sent) for sent in sents]
    return sents

def get_beam_sets(df):
    print("total length is ",len(df))
    best_beam_sets = []
    for i in tqdm(range(len(df))):
        best_set_and_score, _ = beamsearch_selection(prepare_inputs(df.iloc[i]["review_body"]), df.iloc[i]["metareview"], summary_size=100, k=20)
        if best_set_and_score:
            best_beam_sets.append({
                "rouge_score": best_set_and_score[0],
                "sentences": best_set_and_score[1],
            })
        else: 
            print("position is", i, "has value", best_set_and_score)
            best_beam_sets.append({
                "rouge_score": None,
                "sentences": [],
            })
    return best_beam_sets

def add_sentence(df,best_beam_sets):
    print("added sentences length to df ",len(df))
    rouge_score =[]
    selected_sents = []
    for i in tqdm(range(len(df))):
        sents = prepare_inputs(df.iloc[i]["review_body"])
        sents_content = [sents[j] for j in best_beam_sets[i]["sentences"]]
        rouge_score.append(best_beam_sets[i]["rouge_score"])
        selected_sents.append(sents_content)
    print("length of rouge_score",len(rouge_score))
    print("length of selected_summary",len(selected_sents))
    df["selected_rouge"] = rouge_score
    df["selected_sents"] = selected_sents
    return df

df = pd.read_csv("../data/allyears.csv")
year = 2022
new_df = df[df["year"]==year]
# print("totaly number in this year is", len(new_df))
# best_beam_sets = get_beam_sets(new_df)
# with open('../data/df-selected_{}.pkl'.format(year), 'wb') as f:
#     pickle.dump(best_beam_sets, f)

with open('../data/df-selected_{}.pkl'.format(year), "rb") as f:
    best_beam_sets = pickle.load(f)

new_df = add_sentence(new_df,best_beam_sets)
new_df.to_csv("../data/df-selected_{}.csv".format(year))

# with open("../data/df-selected_{}.pkl".format(year), "rb") as f:
#     df_selected = pickle.load(f)
# print(df_selected)