import re
import os
import time
import numpy as np
import pandas as pd
from math import log

def tf_and_idf(query_terms):
    # read in input terms
    input_terms = [re.split('[^a-zA-Z0-9]', word.lower())[0] for word in query_terms]
    # tf and idf components for each term in query
    term_dict = {}
    for term in input_terms:
        # 0 - occurrences of term
        # - 1 - documents term appears in
        # - 2 - titles term appears in
        term_dict[term] = [np.zeros((2, 222)), 0, 0]
    # words per each document array
    doc_word_counts = np.zeros((2, 222))
    doc_frequencies = [None] * 222
    # loop over each index file
    for file_name in os.listdir("index/episodes/"):
        # grab index of file
        file_idx = int(file_name.split("_")[0])
        episode_title = re.split('[^a-zA-Z0-9]', titles.loc[file_idx - 1]["title"].lower())
        open_file = f'index/episodes/{file_name}'
        # open file and read line by line
        words = pd.read_json(open_file, orient='split')
        doc_word_counts[0, file_idx] = sum(words.occurrences)
        doc_word_counts[1, file_idx] = len(episode_title)
        selection = words[words.index.isin(input_terms)]
        for term, row in selection.iterrows():
            # tf value
            term_dict[term][0][0, file_idx] = row["occurrences"]
            # combine windows
            # it appears in the doc
            term_dict[term][1] += 1



    return input_terms, term_dict, doc_word_counts, doc_frequencies


def calculate_bm(query_terms):
    input_terms, term_dict, doc_word_counts, doc_freq = tf_and_idf(query_terms)

    # only use if relevance scores for documents given a query
    ri = 0
    R = 0
    # length normalization (can be set)
    b = 0.25
    # k values can be changed
    # term frequency ignored at lower value and
    # term presence or absence would matter
    k_one = 1.5
    k_two = 500
    # number of documents
    N = 221

    bm_doc_scores = np.zeros((1, 222))
    avdl = np.average(doc_word_counts)
    for file_idx in range(1, 222):
        dl = doc_word_counts[0, file_idx]
        K = k_one * ((1 - b) + (b * (dl / avdl)))
        bm_score = 0
        for term_val in input_terms:
            # docs containing term
            ni = term_dict[term_val][1]
            # frequency of term i
            fi = term_dict[term_val][0][0, file_idx]
            qfi = input_terms.count(term_val)
            term_one_num = (ri + 0.5) / (R - ri + 0.5)
            term_one_den = (ni - ri + 0.5) / (N - ni - R + ri + 0.5)
            term_two_num = (k_one + 1) * fi
            term_two_den = K + fi
            term_three_num = (k_two + 1) * qfi
            term_three_den = k_two + qfi
            bm_i_score = log(term_one_num / term_one_den) * \
                         (term_two_num / term_two_den) * \
                         (term_three_num / term_three_den)
            bm_score += bm_i_score
            # add in multiplier of query terms that appear in the same window in doc
        bm_doc_scores[0, file_idx] = bm_score
    top_ten = bm_doc_scores[0, 1:].argsort()[-10:]
    return top_ten, np.asarray(doc_freq[1:])[top_ten]

# def get_json_string(top_ten, windows=None):
#     titles = pd.read_json("index/seasons.json", orient='records')
#     start = -1
#     end = -11
#
#     json_info = []
#     for doc_id in range(start, end, -1):
#         preview = [''] * 10
#         line_num_ref = 0
#         doc_json = {}
#         doc_num = top_ten[doc_id]
#         episode = titles.loc[doc_num]
#         # build different result if BM25
#         if is_bm:
#             window_value = windows[doc_id].split(" ")[0]
#             words = pd.read_json(f'index/episodes/{doc_num+1}_{episode["href"]}.json', orient='split')
#             first_flag = True
#             for term, row in words.iterrows():
#                 for line_info in row["windows"]:
#                     split_line_info = line_info.split(":")
#                     if window_value == split_line_info[0]:
#                         preview[int(split_line_info[1])] = term
#                         if first_flag:
#                             line_num_ref = split_line_info[2]
#                             first_flag = False
#         title = f'{episode["href"]}.html'
#         with open(f'index/scripts/{title}', 'r') as file:
#             data = file.read()
#         preview = ' '.join(preview)
#         doc_json["episode"] = episode["title"]
#         doc_json["episodeNumber"] = str(doc_num+1)
#         doc_json["link"] = data#f'index/scripts/{episode["href"]}.html#{line_num_ref}'
#         doc_json["line"] = str(line_num_ref)
#         doc_json["preview"] = preview
#         doc_json["season"] = str(episode["season"])
#         json_info.append(doc_json)
#
#
#     return json_info

def pick_metric(mode, query):
    query = query.split(" ")
    if mode == 'bm25':
        start = time.time()
        results, windows = calculate_bm(query)
        end = time.time()
        print(f'BM25: Returned 10 results in {end - start:.2f}s')
        #return get_json_string(results, True, windows)













def main():
    pass


if __name__ == "__main__":
    main()