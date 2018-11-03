# python3.5 build-tagger.py <train_file_absolute_path> <model_file_absolute_path>
# python build-tagger.py sents.train model-file

import os
import math
import sys
import datetime
import numpy as np
import pickle as pk

START = '<s>'
END = '</s>'

# Smoothing
ADD_ONE = 'add_one'
ADD_ONE_CONSTANT = 1
KNESER_NEY = 'kneser_ney'
WITTEN_BELL = 'witten_bell'
KNESER_NEY_DISCOUNT = 0.01
SMOOTHING = KNESER_NEY

def generate_transition_matrix(cij, cj, types, different_types=False, smoothing=None, add_unknown=False):
    '''
    Given counts of ij and j respectively, generate and return transition matrix i --> j
    Resulting cij[i][j] is conditional probability p(i|j).
    '''
    # Count total number of distinct i,j pairs
    distinct_pairs = 0
    for i in cij:
        for j in cij[i]:
            if cij[i][j] > 0:
                distinct_pairs += 1

    # Vocab
    vocab = len(cij.keys())

    # Add UNK to vocab
    if add_unknown is True:
        cij['UNK'] = {}

    # Generate probabilities
    for i in cij:
        if smoothing is WITTEN_BELL:    
            seen_types = count_types(cij[i], seen=True)
            unseen_types = count_types(cij[i], seen=False)
        for j in cj:
            if j not in cij[i]:
                cij[i][j] = 0
            if smoothing is ADD_ONE:
                # Add One smoothing
                cij[i][j] = divide(cij[i][j] + ADD_ONE_CONSTANT, cj[j] * ADD_ONE_CONSTANT + vocab)
            elif smoothing is KNESER_NEY:
                # Kneser Ney smoothing
                if cij[i][j] > 0:
                    cij[i][j] = divide(cij[i][j] - KNESER_NEY_DISCOUNT, cj[j])
                else:
                    cij[i][j] = divide(len(cij[i]), distinct_pairs) * divide(KNESER_NEY_DISCOUNT * types[j], cj[j])
            elif smoothing is WITTEN_BELL:
                # Witten Bell smoothing
                if cij[i][j] > 0:
                    cij[i][j] = divide(cij[i][j], cj[j] + seen_types)
                else:
                    cij[i][j] = divide(seen_types, unseen_types * (cj[j] + seen_types))
            else:
                # No smoothing
                cij[i][j] = divide(cij[i][j], cj[j])
    return cij

def count_types(a, seen=False):
    count_zeroes = 0
    for i in a:
        if a[i] == 0:
            count_zeroes += 1
    if seen is True:
        return len(a) - count_zeroes
    else:
        return count_zeroes

def divide(a, b):
    if b == 0:
        return 0
    return a / float(b)

def add_count(map, key1, key2):
    if key1 in map and key2 in map[key1]:
        map[key1][key2] += 1
    else: 
        if key1 not in map:
            map[key1] = {}
        map[key1][key2] = 1

    return map

def add_total_count(map, key):
    if key in map:
        map[key] += 1
    else:
        map[key] = 1
    return map

def count_words_tags(data):
    '''
    Given a list of sentences, where each sentence is tokenized (delimited by ' ') and each token is tagged
    with its POS tag, we produce two dictionaries w_t and t_t. For each tag, we also count the number of times it occurs
    and return it as tag_counts dictionary.

    w_t['word']['tag'] is the number of times token 'word' appeared in the corpus with pos tag 'tag'
    t_t['tag']['prev_tag'] is the number of times pos tag 'tag' appeared in the corpus after pos tag 'prev_tag'
    word_tag_pairs['tag'] is the number of distinct (word,tag) pairs
    tag_tag_pairs['tag'] is the number of distinct (tag',tag) pairs
    tags is a list of distinct tags found in the corpus
    tag_counts['tag'] is the number of times pos tag 'tag' appeared in the corpus 
    '''
    w_t = {}
    t_t = {}
    tag_counts = {}
    tags = []
    for line in data:
        tokens = line.split(' ')
        prev_tag = START
        tag_counts = add_total_count(tag_counts, prev_tag)
        tags.append(prev_tag)
        for token in tokens:
            token_texts = token.split('/')
            token_tag = token_texts[-1]
            for i in range(len(token_texts) - 1):
                token_text = token_texts[i]

                # add to word,tag and tag,prev_tag,tag counts
                w_t = add_count(w_t, token_text, token_tag)
                t_t = add_count(t_t, token_tag, prev_tag)
                tag_counts = add_total_count(tag_counts, token_tag)

                # add to list of distinct tags
                tags.append(token_tag)

                prev_tag = token_tag
        # add </s>,last_token count
        t_t = add_count(t_t, END, tokens[-1].split('/')[-1])
    
    tags = sorted(list(set(tags)))

    # Count distinct (word,tag) and (tag,_tag) pairs for each tag
    word_tag_pairs = {}
    tag_tag_pairs = {}
    for tag in tags:
        word_tag_pairs[tag] = 0
        for word in w_t:
            if tag in w_t[word]:
                word_tag_pairs = add_total_count(word_tag_pairs, tag)
        tag_tag_pairs[tag] = 0
        for _tag in t_t:
            if tag in t_t[_tag]:
                tag_tag_pairs = add_total_count(tag_tag_pairs, tag)
    return w_t, t_t, word_tag_pairs, tag_tag_pairs, tags, tag_counts

def read_input(fname):
    '''
    Read file line-by-line into list and return
    '''
    with open(fname) as f:
        lines = f.readlines()
    return [x.strip() for x in lines]      

def save_model(wt, tt, tags, outfile):
    model = {
        "word_tag": wt,
        "tag_tag": tt,
        "tags": tags
    }
    with open(outfile, 'wb') as handle:
        pk.dump(model, handle, protocol=pk.HIGHEST_PROTOCOL)

def train_model(train_file, model_file):
    data = read_input(train_file)
    w_t, t_t, word_tag_pairs, tag_tag_pairs, tags, tag_counts = count_words_tags(data)
    p_wt = generate_transition_matrix(w_t, tag_counts, word_tag_pairs, different_types=True, smoothing=SMOOTHING, add_unknown=True)
    p_tt = generate_transition_matrix(t_t, tag_counts, tag_tag_pairs, smoothing=SMOOTHING)
    tags.remove(START)
    save_model(p_wt, p_tt, tags, model_file)
    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
