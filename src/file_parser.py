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


def make_indices():
    # todo will need to update folder location
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer('[^a-zA-Z0-9]', gaps=True)
    start = time.time()
    file_dict = {}
    doc_dict = {}
    offset_dict = {}
    total_docs = 0
    total_words = 0
    for subdir, dirs, file_names in os.walk('../data/'):
        print(f"Processing subdirectory: {subdir}")
        directory = subdir.split("/")[2]
        if not re.match("../data/queries/*", subdir) and directory != '':
            # get the current
            dir_num = int(directory[-2:])
            for file_name in tqdm(file_names):
                # if it's a valid file, open it and read
                if file_name.endswith('.html'):
                    try:
                        with open(f'../data/{directory}/{file_name}', 'r', encoding='utf-8') as html_file:

                            html_in = BeautifulSoup(html_file, 'html.parser', from_encoding='utf-8')
                            offset = html_in.find_all('title')[0].attrs["offset"]
                            # update max doc_id seen for this folder
                            if dir_num in offset_dict.keys():
                                if offset > offset_dict[dir_num]:
                                    offset_dict[dir_num] = offset
                            else:
                                offset_dict[dir_num] = offset

                            file_identifier = f'{dir_num}-{offset}'
                            text = html_in.contents[0].get_text()
                            # tokenize text
                            nltk_tokenized = tokenizer.tokenize(text)
                            # find total words in the document
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
            if subdir != "../data/" and directory != "queries":
                # save all sub index files in case of error during indexing
                with open(f'../index/{directory}', 'w+', encoding='utf-8') as index_file:
                    for word, file_iden_dict in file_dict.items():
                        second_col = []
                        total_occ = 0
                        for file_name_off, count in file_iden_dict.items():
                            total_occ += count
                            total_offset = 0
                            file_split = file_name_off.split("-")
                            folder = int(file_split[0])-1
                            doc_id = int(file_split[1])
                            if folder > 0:
                                while folder > 0:
                                    total_offset += offset_dict[folder]
                                    folder -= 1
                                    total_offset += folder
                            delta_encode = doc_id + total_offset
                            second_col.append(f'{delta_encode}:{count}')
                        second_col.sort(key=lambda x: int(x.split(":")[0]))
                        index_file.write(f'{word}\t{total_occ}\t{second_col}\n')

    avg_doc_length = math.floor(total_words / total_docs)
    with open(f'../index/doc_info.idx', 'w+', encoding='utf-8') as doc_file:
        doc_file.write(f'{total_docs}\t{avg_doc_length}\t0\n')
        for doc_file_name, count in doc_dict.items():
            # offset, filename, word count
            doc_file.write(f'{doc_file_name}\t{count[0]}\t{count[1]}\n')

    with open(f'../index/folder_info.idx', 'w+', encoding='utf-8') as folder_info:
        for folder_id, max_val in offset_dict.items():
            folder_info.write(f'{folder_id}\t{max_val}')

    with open(f'../index/main.idx', 'w+', encoding='utf-8') as index_file:
        for word, file_iden_dict in file_dict.items():
            second_col = []
            total_occ = 0
            for file_name_off, count in file_iden_dict.items():
                total_occ += count
                total_offset = 0
                file_split = file_name_off.split("-")
                folder = int(file_split[0]) - 1
                doc_id = int(file_split[1])
                if folder > 0:
                    while folder > 0:
                        total_offset += offset_dict[folder]
                        folder -= 1
                        total_offset += folder
                delta_encode = doc_id + total_offset
                second_col.append(f'{delta_encode}:{count}')
            second_col.sort(key=lambda x: int(x.split(":")[0]))
            index_file.write(f'{word}\t{total_occ}\t{second_col}\n')

    end = time.time()
    # TODO delete sub index files so they dont take up space
    print(f"\nCompleted Walkthrough of data directory!: {end - start:.2f}s")


def main():
    make_indices()


if __name__ == "__main__":
    main()
