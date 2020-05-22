import json
import re
import os
import nltk
from nltk.tokenize import WordPunctTokenizer
from collections import defaultdict
import numpy as np

'''
This script helps build a trainable corpus using the COVID-19 Open Research
Dataset Challenge (https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge). 
This dataset is a collection of COVID related articles.

We process the dataset in the following way:
    1. Check if certain related keywords metadata associated with each 
        article to ensure we are only grabbing COVID-19 articles.
    2. We read in a total of MAX_FILES articles which are split into training,
        testing, and validation sets.
    3. We process each individual article by tokenizing the article (using nltk) and
        removing single characters (excluding a and i) and infrequent tokens.
'''

# Collection of relevant terms that will help find COVID-19 articles
key_words = ["COVID-19", "Coronavirus 19", "Coronavirus-19", "COVID 19", "SARS-CoV-2"]
key_words = list(map(str.upper, key_words))
covid_words = re.compile("|".join(key_words))

# Maximum number of COVID 19 files we process
MAX_FILES = 5000

# Minimum number of occurance for a word
TOKEN_FREQUENCY = 3

TRAIN_SPLIT = .8
VALIDATION_SPLIT = 0
TEST_SPLIT = .2

def consolidate(meta_data):
    '''
    This function wil merge all of the tokens of each file and some additional cleaning.
    '''
    # merge all of the tokens
    tokens = []
    for file in meta_data:
        tokens += meta_data[file]
        
    freq = defaultdict(lambda : 0)
    cleaned_tokens = []
    
    # Remove single character count freqs
    for index, token in enumerate(tokens):
        if not (len(token) <= 1 and not token.isalnum() and token not in ['a', 'i']):
            cleaned_tokens.append(token)
            freq[token] += 1

    tokens = cleaned_tokens
    tokens = list(filter(lambda token: freq[token] >= TOKEN_FREQUENCY, tokens))
    return tokens

def clean(body):
    '''
    Helper function where we can clean the text of the data in the way we want.
    Should return the body as well as the tokens in the body.
    '''
    sentences = nltk.sent_tokenize(body)
    tokens = []
    if sentences:
        for index, sentence in enumerate(sentences):
            tokenizer = WordPunctTokenizer()
            sentence_tokens = tokenizer.tokenize(sentence)

            sentence_tokens.insert(0, '<s>')
            sentence_tokens.append('</s>')
            tokens += sentence_tokens
    return tokens

def process(max_files, train_split, test_split, validation_split):
    '''
    Function will build and collect the text and metadata associated with the 
    COVID dataset will use to train.
    '''
    # metadata is {file: tokens} 
    test_data = {}
    train_data = {}
    validation_data = {}
    
    # Dataset is directly outside of this folder
    base_path = "../CORD-19-research-challenge/document_parses/pdf_json/"
    
    for path in os.listdir(base_path):
        # Keep processing till we have enough data
        if max_files < 0:
            # Writing the text
            with open('corpus.json', 'w') as fp:
                
                train_tokens = consolidate(train_data)
                test_tokens = consolidate(test_data)
                validation_tokens = consolidate(validation_data)
                
                corpus = {'train_count' : len(train_tokens), 'train_tokens' : train_tokens, 'test_count' : len(test_tokens), 'test_tokens' : test_tokens, 'validation_count' : len(validation_tokens), 'validation_tokens' : validation_tokens}
                json.dump(corpus, fp)
        
            return
        
        # I/O
        file = open(base_path + path)
        text = json.load(file)
        file.close()

        # Check if we have a match in the metadata of the article
        if covid_words.search(text['metadata']['title'].upper()):
            max_files -= 1
            if max_files % 100 == 0:
                print(f"Currently there are {max_files} files left to process.")
            
            # The body of text is always in list of texts
            raw = " ".join([content['text'] for content in text['body_text']])
            
            # Clean the body of text and return the tokens
            tokens = clean(raw)
            
            prob = np.random.uniform()
            
            if prob <= train_split:
                train_data[path] = tokens
            elif train_split < prob <= 1 - validation_split:
                test_data[path] = tokens
            else:
                validation_data[path] = tokens

    # Writing the text
    with open('corpus.json', 'w') as fp: 
        train_tokens = consolidate(train_data)
        test_tokens = consolidate(test_data)
        validation_tokens = consolidate(validation_data)

        corpus = {'train_count' : len(train_tokens), 'train_tokens' : train_tokens, 'test_count' : len(test_tokens), 'test_tokens' : test_tokens, 'validation_count' : len(validation_tokens), 'validation_tokens' : validation_tokens}
        json.dump(corpus, fp)
        

process(max_files=MAX_FILES, train_split=TRAIN_SPLIT, test_split=TEST_SPLIT, validation_split=VALIDATION_SPLIT)
