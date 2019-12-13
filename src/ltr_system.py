import pandas as pd
from math import log
import time
# TODO page 312, 178 for skip pointers
# TODO find same set between all three, if over 10, calculate that

def merge_lists(selection):
    file_to_word_df = pd.read_csv('../index/scores.idx', sep='\t', header=None).set_index(0, drop=True)
    file_to_word = file_to_word_df[1].to_dict()
    for term, row in selection.iterrows():
        appear_list = eval(row[2][10:-1])
        for doc_id in appear_list:
            file_to_word[doc_id] += 1
    return pd.DataFrame.from_dict(file_to_word, orient='index').sort_values(by=[0], ascending=False)[:5000]

def prob_word_doc(selection, mu_val_set):
    if selection.shape[0] > 1:
        top_5000 = merge_lists(selection)
        try_top = list(top_5000.index.values)
    else:
        try_top = eval(selection[2][0][10:-1])
    doc_info_csv = pd.read_csv("../index/doc_info.idx", sep='\t', header=None)

    # between 0-1
    # lambda_val = 1
    mu_val = mu_val_set
    # number of documents
    N = int(doc_info_csv.loc[0][0])
    # average document length
    avdl = int(doc_info_csv.loc[0][1])
    # total word occurrences in collection
    collection_c = N * avdl

    word_to_doc = {}

    # only look at top 5000 docs
    doc_info_csv = doc_info_csv.set_index(1, drop=True).drop(columns=0)
    doc_info_csv = doc_info_csv.loc[try_top]
    file_to_word_df = pd.read_csv('../index/scores.idx', sep='\t', header=None).set_index(0, drop=True)
    file_to_word = file_to_word_df.loc[try_top][1].to_dict()

    for term, row in selection.iterrows():
        word_to_doc[term] = dict(zip(eval(row[2][10:-1]), eval(row[3][12:-1])))

    for term, row in selection.iterrows():
        # how many times a word appears in collection (sum of appears column)
        c_qi = int(row[1])
        for doc_id in try_top:
            try:
                # frequency of term in document
                f_qi_d = word_to_doc[term][doc_id]
            except KeyError:
                f_qi_d = 0
            # length of document
            dl = doc_info_csv.loc[doc_id][2]
            # frequency of term in document + mu value * term occurrences
            # in collection over total collection size
            numerator = f_qi_d + (mu_val*(c_qi/collection_c))
            denominator = dl + mu_val
            t_i_score = log(numerator/denominator)
            file_to_word[doc_id] += t_i_score
    results = pd.DataFrame.from_dict(file_to_word, orient='index')
    x_row = results.sort_values(by=[0], ascending=False)[:10]
    return x_row
