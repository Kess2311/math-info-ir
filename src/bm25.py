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
from src.ltr_system import prob_word_doc

def tf_and_idf(query):
    start = time.time()
    # read in input terms
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer('[^a-zA-Z0-9]', gaps=True)
    query_tokens = tokenizer.tokenize(query)
    input_terms = []
    for term in query_tokens:
        if term not in stop_words:
            input_terms.append(term.lower())

    # open file and read line by line
    words = pd.read_csv("../index/main.idx", sep='\t', header=None, index_col=0)
    words = words.sort_values(by=[1], ascending=False)
    selection = words[words.index.isin(input_terms)]
    end = time.time()
    print(f'tf&idf: {end - start:.2f}s')
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

    # TODO make dict with title -> score
    # send to Pandas and sort

    doc_score_dict = {}
    for term, row in selection.iterrows():
        doc_appear_list = dict(zip(eval(row[2][10:-1]), eval(row[3][12:-1])))
        for doc in doc_appear_list.keys():
            dl = doc_info_csv[doc_info_csv[1] == doc][2].values[0]
            K = k_one * ((1 - b) + (b * (dl / avdl)))
            # docs containing term
            ni = len(doc_appear_list)
            # frequency of term i
            fi = doc_appear_list[doc]
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
            if doc in doc_score_dict.keys():
                doc_score_dict[doc] += bm_i_score
            else:
                doc_score_dict[doc] = bm_i_score


    results = pd.DataFrame.from_dict(doc_score_dict, orient='index').sort_values(by=[0], ascending=False)[:10]
    return results

def get_json_string(top_ten):
    title_doc = pd.read_csv("../index/doc_info.idx", sep='\t', header=None)

    result = 1
    for title, value in top_ten.iterrows():
        ans = title_doc[title_doc[1] == title][0].values[0]
        print(f"{result}.) {ans}\t{title}\t{value.values[0]}")
        result += 1

def output_qrels(query_num, method, top_ten, mu=None):
    rank = 1
    if mu:
        file_name = f'../results/q{query_num}_{method}_{mu}.results'
    else:
        file_name = f'../results/q{query_num}_{method}.results'
    with open(file_name, 'w+') as qrel_file:
        for title, value in top_ten.iterrows():
            qrel_file.write(f'{query_num}\tQ0\t{title}\t{rank}\t{value.values[0]}\tdefault\n')
            rank += 1


def pick_metric(mode, query, query_num):
    if mode == 'bm25':
        start = time.time()
        results = calculate_bm(query)
        end = time.time()
        print(f'Query: "{query}"\nBM25: Returned 10 results in {end - start:.2f}s')
        get_json_string(results)
        output_qrels(query_num, 'bm25', results)
    elif mode == 'qlds':
        mu_val = 3500
        start = time.time()
        returned_query, selection = tf_and_idf(query)
        results = prob_word_doc(selection, mu_val)
        end = time.time()
        print(f'Query: "{query}"\n\tTokenized as: {returned_query}\n\tMU_VAL: {mu_val}\nQuery Likelihood with Dirichlet Smoothing: Returned 10 results in {end - start:.2f}s')
        get_json_string(results)
        output_qrels(query_num, 'qlds', results, mu_val)


if __name__ == "__main__":
    query_num = 31

    with open('../data/queries/queries.txt', 'r') as query_file:
        for query in query_file:
            pick_metric("bm25", query, query_num)
            pick_metric('qlds', query, query_num)
            query_num += 1