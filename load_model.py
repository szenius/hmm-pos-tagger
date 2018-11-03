import pickle as pk
with open('model-file', 'rb') as handle:
    model = pk.load(handle)

wt = model['word_tag']
tags = model['tags']

for tag in tags:
    sum = 0
    for word in wt:
        sum += wt[word][tag]
    print(tag, sum)
