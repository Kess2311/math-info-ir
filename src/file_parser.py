import os
import pandas as pd
from tqdm import tqdm
import time
import nltk
import math
import re
nltk.download('stopwords')
nltk.download('punkt')
from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords


def make_zero_doc_file():
    # todo will need to update folder location
    score_dict = {}
    files_df = pd.read_csv("../index/doc_info.idx", sep='\t', header=None)
    for title, row in files_df[1:].iterrows():
        score_dict[row[1]] = 0

    scores_df = pd.DataFrame.from_dict(score_dict, orient='index')
    scores_df.to_csv('../index/scores.idx', sep='\t', header=None)



def make_indices():
    # todo will need to update folder location
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer('[^a-zA-Z0-9]', gaps=True)
    start = time.time()
    file_dict = {}
    doc_dict = {}
    total_docs = 0
    total_words = 0
    for subdir, dirs, file_names in os.walk('../data/'):
        print(f"Processing subdirectory: {subdir}")
        directory = subdir.split("/")[2]
        if not re.match("../data/queries/*", subdir) and directory != '':
            # get the current
            dir_num = int(directory[-2:])
            for file_name in tqdm(file_names):
                if file_name.endswith('.html'):
                    try:
                        with open(f'../data/{directory}/{file_name}', 'r', encoding='utf-8') as html_file:
                            html_in = BeautifulSoup(html_file, 'html.parser', from_encoding='utf-8')
                            offset = html_in.find_all('title')[0].attrs["offset"]
                            file_identifier = f'{dir_num}-{offset}'
                            text = html_in.contents[0].get_text()
                            nltk_tokenized = tokenizer.tokenize(text)
                            doc_length = len(nltk_tokenized)
                            total_words += doc_length
                            doc_dict[file_name] = [file_identifier, doc_length]
                            total_docs += 1
                            # tokenize all words with a regex tokenizer and lower case
                            for word in nltk_tokenized:
                                lower = word.lower()
                                if lower not in stop_words:
                                    if lower in file_dict.keys():
                                        if file_identifier in file_dict[lower].keys():
                                            file_dict[lower][file_identifier] += 1
                                        else:
                                            file_dict[lower][file_identifier] = 1
                                    else:
                                        file_dict[lower] = {file_identifier: 1}
                            math_tags = html_in.find_all("math")
                            # run through all math equation tokens without modifying
                            for tag in math_tags:
                                text = word_tokenize(tag.get_text())
                                for word in text:
                                    if word not in stop_words:
                                        if word in file_dict.keys():
                                            if file_identifier in file_dict[word].keys():
                                                file_dict[word][file_identifier] += 1
                                            else:
                                                file_dict[word][file_identifier] = 1
                                        else:
                                            file_dict[word] = {file_identifier: 1}
                    except FileNotFoundError:
                        print(f'File {file_name} not found. Skipping')
                        continue


    avg_doc_length = math.floor(total_words/total_docs)
    with open(f'../index/doc_info.idx', 'w+', encoding='utf-8') as doc_file:
        doc_file.write(f'{total_docs}\t{avg_doc_length}\t0\n')
        for doc_file_name, count in doc_dict.items():
            # offset, filename, word count
            doc_file.write(f'{doc_file_name}\t{count[0]}\t{count[1]}\n')

    with open(f'../index/main.idx', 'w+', encoding='utf-8') as index_file:
        for word, file_iden_dict in file_dict.items():
            total_occ = sum(file_iden_dict.values())
            index_file.write(f'{word}\t{total_occ}\t{file_iden_dict.keys()}\t{file_iden_dict.values()}\n')
    make_zero_doc_file()
    end = time.time()
    print(f"\nCompleted Walkthrough of data directory!: {end - start:.2f}s")


def main():
    make_indices()


if __name__ == "__main__":
    main()
