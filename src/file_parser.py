import os
from tqdm import tqdm
import time
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

def make_indices():
    # todo will need to update folder location
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer('[^a-zA-Z0-9]', gaps=True)
    start = time.time()
    for subdir, dirs, file_names in os.walk('../data/'):
        print(f"Processing subdirectory: {subdir}")
        for file_name in tqdm(file_names):
            if file_name.endswith('.html'):
                with open(f'../data/{subdir}/{file_name}', 'r', encoding='utf-8') as html_file:
                    file_dict = {}
                    html_in = BeautifulSoup(html_file, 'html.parser', from_encoding='utf-8')
                    offset = html_in.find_all('title')[0].attrs["offset"]
                    text = html_in.get_text()
                    nltk_tokenized = tokenizer.tokenize(text)
                    for word in nltk_tokenized:
                        lower = word.lower()
                        if lower not in stop_words:
                            if lower in file_dict.keys():
                                file_dict[lower] += 1
                            else:
                                file_dict[lower] = 0
                    with open(f'../index/{offset}.json', 'w+', encoding='utf-8') as index_file:
                        for word, freq in file_dict.items():
                            index_file.write(f'{word}\t{freq}\n')
    end = time.time()
    print(f"\nCompleted Walkthrough of data directory!: {start-end:.2f}")

def main():
    make_indices()


if __name__ == "__main__":
    main()