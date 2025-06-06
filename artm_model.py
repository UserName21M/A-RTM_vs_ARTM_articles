# %%

import artm
import numpy as np
import pickle

PATH = 'dataRU/data40.json_'

with open(PATH + 'combinations', 'rb') as file:
    combinations = pickle.load(file)

with open('stem2word', 'rb') as file:
    stem2word = pickle.load(file)

# %%

bv = artm.BatchVectorizer(data_path = PATH + 'data_vw',
                          data_format = 'vowpal_wabbit',
                          batch_size = 100,
                          target_folder = 'batches',
                          class_ids = {'@default_class': 1.0, '@title_class': 5.0} )
dictionary = artm.Dictionary()
dictionary.gather(data_path = 'batches', vocab_file_path = PATH + 'vocab_vw')

# %%

model = artm.ARTM(num_topics = 100, num_document_passes = 10, dictionary = dictionary)

model.scores.add(artm.TopicKernelScore(name = 'kernels', probability_mass_threshold = 0.1))
model.scores.add(artm.PerplexityScore(name = 'perplexity', dictionary = dictionary))
model.scores.add(artm.TopTokensScore(name = 'top-tokens', num_tokens = 10))
model.scores.add(artm.SparsityPhiScore(name = 'sparsity'))

model.regularizers.add(artm.DecorrelatorPhiRegularizer(name = 'decorrelator', tau = 7e1))

for i in range(20):
    model.fit_offline(bv)
    perplexity = model.score_tracker['perplexity'].last_value
    sparsity = model.score_tracker['sparsity'].last_value
    print(f'Epoch {i} | Perplaxity = {perplexity}, Sparsity = {sparsity}')
top_tokens = model.score_tracker['top-tokens'].last_tokens

# theta = model.transform(batch_vectorizer = bv, theta_matrix_type = 'dense_theta').sort_index(axis = 1).to_numpy()
# phi = model.get_phi_dense()[0]

topic_names = [i for i in model.topic_names[1:] if i in top_tokens]

# %%

def change_word(x : str):
    if x in stem2word:
        return stem2word[x]
    return x

for topic in topic_names:
    print(topic, *map(change_word, top_tokens[topic]))

# %%

total_words = 0
for word in combinations:
    total_words += combinations[word]['APPEARANCES']

def PPMI(w1 : str, w2 : str):
    if not w2 in combinations[w1]:
        c = 0
    else:
        c = combinations[w1][w2]
    return np.max(np.log(c * total_words / combinations[w1]['APPEARANCES'] / combinations[w2]['APPEARANCES'] + 1e-8), 0)

def coherence(words : list):
    total = 0
    ln = len(words)

    for i in range(ln - 1):
        word1 = words[i]
        for j in range(i + 1, ln):
            word2 = words[j]
            total += PPMI(word1, word2)

    if ln <= 1:
        return 0
    return total * 2 / ln / (ln - 1)

# %%

size = model.score_tracker['kernels'].last_size
purity = model.score_tracker['kernels'].last_purity
contrast = model.score_tracker['kernels'].last_contrast
coherences = list(coherence(top_tokens[i]) for i in topic_names)

# %%
