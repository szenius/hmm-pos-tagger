# python3.5 run-tagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>
# python run-tagger.py sents.test model-file sents.out

import os
import math
import sys
import datetime
import numpy as np
import pickle as pk
from math import log

START = '<s>'
END = '</s>'

def add_pos_tags(test_data, p_wt, p_tt, tags):
    '''
    Runs Viterbi Algorithm on each line of the test_data to add POS tags.
    Returns POS tagged test_data.
    '''
    tagged = []

    for line in test_data:
        tokens = line.split(' ')

        ### Viterbi Algorithm ###
        v = np.zeros(shape=(len(tags), len(tokens)), dtype=float)
        chosen = np.zeros(shape=(len(tags), len(tokens)), dtype=float)
        # Forward step to calculate all v values
        for i in range(len(tokens) - 1):
            token = tokens[i]
            token = token.split('/')[0]
            for j in range(len(tags)):
                tag = tags[j]
                if i == 0:
                    # first layer
                    v[j, i] = product(retrieve_prob(p_tt, tag, START), retrieve_prob(p_wt, token, tag, check_unknown=True))
                else:
                    # middle layers
                    max_prob, chosen_prev_tag = retrieve_max(v[:,i-1], tag, p_tt, tags)
                    v[j,i] = product(max_prob, retrieve_prob(p_wt, token, tag, check_unknown=True))
                    chosen[j,i] = chosen_prev_tag
        # Backtrack to get viterbi trace and tag tokens
        max_prob, last_chosen_tag = retrieve_max(v[:,i-1], END, p_tt, tags)
        for i in range(len(tokens) - 1, -1, -1):
            tokens[i] += "/" + tags[last_chosen_tag]
            last_chosen_tag = int(chosen[last_chosen_tag,i])
            
        # Add tagged line to result
        tagged.append(' '.join(tokens))

    return tagged

def product(p1, p2, log_sum=False):
    if log_sum is False:
        return p1 * p2
    else:
        return log(p1 + 1e-20, 10) + log(p2 + 1e-20, 10)

def retrieve_max(prev_layer, tag, p_tt, tags):
    '''
    Compute max{v_(i-1)(j) * p_tt[tag][tags_list[j]] for each j}. Return the max value and j.

    prev_layer: v[:,i-1]
    tag: current tag
    p_tt: p(tag|prev_tag)
    tags: list of all tags
    '''
    max = 0
    chosen = 0
    for j in range(len(tags)):
        prev_tag = tags[j]
        val = product(prev_layer[j], retrieve_prob(p_tt, tag, prev_tag), log_sum=False)
        if val > max:
            max = val
            chosen = j
    return max, chosen

def retrieve_prob(map, key1, key2, check_unknown=False):
    if key1 in map and key2 in map[key1]:
        return map[key1][key2]
    elif check_unknown is True:
        return map['UNK'][key2]
    else:
        print("Warning:", key1, key2, "zero probability!")
        return 0

def write_results(tagged, out_file):
    with open(out_file, 'w') as f:
        for line in tagged:
            f.write("%s\n" % line)

def load_model(infile):
    '''
    Loads model from model file and returns the p(word|tag) and p(tag|prev_tag) probabilities.
    '''
    with open(infile, 'rb') as handle:
        model = pk.load(handle)
    return model['word_tag'], model['tag_tag'], model['tags']

def read_input(fname):
    '''
    Read file line-by-line into list and return
    '''
    with open(fname) as f:
        lines = f.readlines()
    return [x.strip() for x in lines]

def tag_sentence(test_file, model_file, out_file):
    p_wt, p_tt, tags = load_model(model_file)
    test_data = read_input(test_file)
    tagged = add_pos_tags(test_data, p_wt, p_tt, tags)
    write_results(tagged, out_file)
    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
