# %% word2vec
# variant: skip gram with negative sampling
# %%
import numpy as np
from collections import Counter
import random
# %% Preprocessing

with open("text8", "r", encoding="utf-8") as f:
    text = f.read()

words = text.split()

word_freqs = Counter(words)

min_count = 5 # discarding all the words that appear less than min_count times in text
words = [word for word in words if word_freqs[word] >= min_count]
N = len(words)
word_freqs = Counter({word: cnt for word, cnt in word_freqs.items() if cnt >= min_count})
words_unique = list(word_freqs.keys())
V = len(words_unique) # number of different words in the text

freqs = np.array(list(word_freqs.values()), dtype="float")
freqs /= freqs.sum() # normalize, get actual frequencies(probabilities)

t = 1e-5
central_word_prob = np.minimum(1.0,np.sqrt(t/freqs)+t/freqs)
central_word_prob_d = dict(zip(words_unique,central_word_prob))
# probability of not discarding a specific occurence of word as central

neg_sampling_word_prob = freqs**0.75
neg_sampling_word_prob /= neg_sampling_word_prob.sum() # again, normalize
neg_sampling_word_prob_l = list(neg_sampling_word_prob)
# distribution needed for negative sampling

word_to_idx = {w:i for i,w in enumerate(words_unique)}
# needed for faster reaching of rows and columns of weight matrices

d = 100 # number of nodes in hidden layer (the number of dimensions in a space of vectors the words live in)
K = 5 # number of words for negative sampling
m = 3 # window radius
print('done')
# %% Initialization of weights
rng = np.random.default_rng()
W_in = rng.uniform(-0.5/d, 0.5/d, (V,d))
W_out = rng.uniform(-0.5/d, 0.5/d, (d,V))
# %% Funcions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_window(words, cent, radius):
    low = max(cent - radius, 0)
    high = min(cent + radius + 1, len(words) - 1)
    window = [word for word in words[low:high] if word != words[cent]]
    # remove occurrences of the exact word w to get a final window

    return window

def negative_sampling(word_in, word_out, word_bank, weights, K):
    neg_sampled_words = random.choices(word_bank, weights=weights, k=K)
    while (word_in in neg_sampled_words) or (word_out in neg_sampled_words):
        neg_sampled_words = random.choices(word_bank, weights=weights, k=K)

    return neg_sampled_words

def get_indexes(word_to_idx, word_in, word_out,neg_sampled_words):
    in_idx = word_to_idx[word_in]
    out_idx = word_to_idx[word_out]
    neg_samp_idxs = np.array([word_to_idx[w] for w in neg_sampled_words], dtype=int)

    return in_idx, out_idx, neg_samp_idxs

def monitoring(i,W_in,Loss):
    print('iteration: ',i)
    Loss
    print('Average loss: ', Loss)
    max_norm = np.max(np.linalg.norm(W_in,axis=1))
    print('Max embedding norm: ',max_norm)
    print('________________________________')
# %% Training loop

max_iters = N # can be smaller number if we want to train on a part of the text
epochs = 3 # number of epochs
lr0 = 0.025 # starting learning ratio
mon_period = 1e+3 # monitoring is done once in every mon_period iterations
Loss = 0 # Loss that will be used for monitoring
for ep in range(epochs):
    i = -1
    used = 0 # for averaging Loss
    for w in words:
        i += 1
        if i>=max_iters:
            break
        if (i>0) and (i%mon_period == 0): # monitoring done once in every 1000 central words
            Loss /= used
            monitoring(i,W_in,Loss)
            Loss = 0
            used = 0
        cont_flag = rng.binomial(n=1,p=central_word_prob_d[w]) # skipping central word with some probability
        if cont_flag==0:
            continue

        lr = lr0 * (1 - (i + ep * max_iters) / (epochs * max_iters))
        window = get_window(words, i, m)
        for wo in window:

            # negative sampling
            neg_sampled_words = negative_sampling(w,wo,words_unique,neg_sampling_word_prob_l,K)

            # determining rows and columns of matrices that are going to be changed
            in_idx, out_idx, neg_samp_idxs = get_indexes(word_to_idx, w, wo, neg_sampled_words)

            # W_in size is (V,d)
            # W_out size is (d,V)
            # size of the following vectors is (d,)
            v_i = W_in[in_idx, :]
            u_o = W_out[:,out_idx]
            u_nk = W_out[:, neg_samp_idxs]

            #Loss += -ln(sigmoid(u_o.T*v_i) -sum(1,K){ln(sigmoid(-u_nk.T * v_i))}
            Loss += np.logaddexp(0,-np.dot(u_o,v_i)) + np.sum(np.logaddexp(0,np.dot(u_nk.T,v_i)))
            # accumulating Loss for monitoring

            dL_dvi = -(1-sigmoid(np.dot(u_o,v_i)))*u_o # gradient component for input
            for j in range(K):
                dL_dvi += sigmoid(np.dot(u_nk[:,j],v_i))*u_nk[:,j]
            W_in[in_idx,:] -= lr*dL_dvi.T

            dL_duo = -(1-sigmoid(np.dot(u_o,v_i)))*v_i # gradient component for output
            W_out[:,out_idx] -= lr*dL_duo

            for j in range (K):
                dL_dunk = sigmoid(np.dot(u_nk[:,j],v_i))*v_i # gradient components for negative sampled words
                W_out[:,neg_samp_idxs[j]] -= lr*dL_dunk
            #weight updates
        used += len(window)

    print('Epoch ',ep,' done.')

#%% saving
    np.savez(
        "word2vec_model.npz",
        W_in=W_in,
        W_out=W_out,
        words_unique=np.array(words_unique)
    )