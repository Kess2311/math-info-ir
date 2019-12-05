import re
import os
import time
import numpy as np
import pandas as pd
from math import log
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

def tf_and_idf(query):
    # read in input terms
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer('[^a-zA-Z0-9]', gaps=True)
    query_tokens = tokenizer.tokenize(query)
    input_terms = []
    for term in query_tokens:
        if term not in stop_words:
            input_terms.append(term.lower())
    # tf and idf components for each term in query
    term_dict = {}
    for term in input_terms:
        # 0 - doc_num
        # 1 - total_occ in doc
        term_dict[term] = []

    # open file and read line by line
    words = pd.read_csv("../index/main.idx", sep='\t', header=None, index_col=0)
    words = words.sort_values(by=[1], ascending=False)
    selection = words[words.index.isin(input_terms)]

    return input_terms, selection


def calculate_bm(query_terms):
    input_terms, selection = tf_and_idf(query_terms)
    doc_info_csv = pd.read_csv("../index/doc_info.idx", sep='\t', header=None)
    # only use if relevance scores for documents given a query
    ri = 0
    R = 0
    # length normalization (can be set)
    b = 0.75
    # k values can be changed
    # term frequency ignored at lower value and
    # term presence or absence would matter
    k_one = 1.5
    k_two = 500
    # number of documents
    N = int(doc_info_csv.loc[0][0])

    avdl = int(doc_info_csv.loc[0][1])

    doc_score_dict = {}
    for idx, row in doc_info_csv.iterrows():
        doc_score_dict[row[1]] = 0
    for term, row in selection.iterrows():
        doc_appear_list = eval(row[1])
        for doc in doc_appear_list:
            split_val = doc.split(":")
            doc_name = split_val[0]
            dl = doc_info_csv[doc_info_csv[1] == doc_name][2].values[0]
            K = k_one * ((1 - b) + (b * (dl / avdl)))
            # docs containing term
            ni = len(doc_appear_list)
            # frequency of term i
            fi = int(split_val[1])
            qfi = input_terms.count(term)
            term_one_num = (ri + 0.5) / (R - ri + 0.5)
            term_one_den = (ni - ri + 0.5) / (N - ni - R + ri + 0.5)
            term_two_num = (k_one + 1) * fi
            term_two_den = K + fi
            term_three_num = (k_two + 1) * qfi
            term_three_den = k_two + qfi
            bm_i_score = log(term_one_num / term_one_den) * \
                         (term_two_num / term_two_den) * \
                         (term_three_num / term_three_den)
            doc_score_dict[doc_name] += bm_i_score


    results = pd.DataFrame.from_dict(doc_score_dict, orient='index').sort_values(by=[0], ascending=False)[:10]

    return results

def get_json_string(top_ten):
    title_doc = pd.read_csv("../index/doc_info.idx", sep='\t', header=None)

    result = 1
    for title, value in top_ten.iterrows():
        ans = title_doc[title_doc[1] == title][0].values[0]
        print(f"{result}.) {ans}\t{value.values[0]}")
        result += 1



def pick_metric(mode, query):
    if mode == 'bm25':
        start = time.time()
        results = calculate_bm(query)
        end = time.time()
        print(f'Query: "{query}"\nBM25: Returned 10 results in {end - start:.2f}s')
        get_json_string(results)




if __name__ == "__main__":
    pick_metric("bm25", "series")
    pick_metric("bm25", "economic theories and models")
    pick_metric("bm25", "when is the letter β used")
    pick_metric("bm25", "Quantum Computing Algorithms")
    pick_metric("bm25", "what is the hot chocolate effect")









def main():
    pass


if __name__ == "__main__":
    main()