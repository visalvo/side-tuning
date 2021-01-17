from __future__ import division, print_function

import numpy as np
import os
import random
import torch
import torch.nn.functional as F
# from torchtext.vocab import FastText, GloVe
from tqdm.notebook import tqdm
from PIL import Image, UnidentifiedImageError
from torch.backends import cudnn
from gensim.models import Word2Vec
from warnings import filterwarnings

# from gensim.models import FastText  # to change

filterwarnings("ignore")
cudnn.deterministic = True
cudnn.benchmark = False

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

nlp_seq_lenght = 100


# convert snippets to tokens
def tokenize_snippet(nlp, snippet):
    tokenized_snippet = []

    for word in snippet:
        try:
            idx = nlp.vocab[word].index
        except:
            idx = 0
        tokenized_snippet.append(idx)

    return tokenized_snippet


'''
# convert snippets to tokens
def tokenize_snippet(nlp, snippet):
    tokenized_snippet = []

    for word in snippet:
        try:
            idx = torch.tensor(nlp[word])
        except:
            idx = torch.zeros(300)

        tokenized_snippet.append(idx)

    return tokenized_snippet
'''


def load_txt_samples(orig_dir, dest_dir, nlp):
    for label in tqdm(sorted(os.listdir(orig_dir)), desc='txts samples'):
        class_path = f'{orig_dir}/{label}'
        with os.scandir(class_path) as it:
            for _, path in tqdm(enumerate(it), leave=False):
                with open(path, 'rb') as f:
                    txt = f.read()
                doc = [''.join([i for i in token.decode('UTF-8') if i.isalnum()]) for token in txt.split()]

                # word2vec = [torch.tensor(nlp[i]) for i in doc]
                # word2vec = [nlp[i] for i in doc]
                word2vec = tokenize_snippet(nlp, doc)

                padding = nlp_seq_lenght - len(word2vec)

                if padding > 0:
                    if padding == nlp_seq_lenght:
                        # x = torch.zeros((nlp_seq_lenght, 300))
                        x = torch.LongTensor(word2vec)
                    else:
                        # x = F.pad(torch.stack(word2vec), [0, 0, 0, padding]) #to check if stack of tensors is right
                        # x = F.pad(torch.tensor(word2vec), [0, 0, 0, padding])
                        x = F.pad(torch.LongTensor(word2vec), [padding, 0])
                else:
                    # x = torch.stack(word2vec[:nlp_seq_lenght])
                    # x = torch.tensor(word2vec[:500])
                    x = torch.LongTensor(word2vec[:nlp_seq_lenght])

                if not os.path.exists(f'{dest_dir}/{label}'):
                    os.mkdir(f'{dest_dir}/{label}')
                torch.save(x.half(), f'{dest_dir}/{label}/{"".join(path.name.split(".")[:-1])}.ptr')


def load_img_samples(orig_dir, dest_dir):
    for label in tqdm(sorted(os.listdir(orig_dir)), desc='imgs samples'):
        class_path = f'{orig_dir}/{label}'
        with os.scandir(class_path) as it:
            for _, path in tqdm(enumerate(it), leave=False):
                with open(path, 'rb') as f:
                    try:
                        img = Image.open(f)
                        img = img.convert('RGB')
                        img = img.resize((384, 384))
                        if not os.path.exists(f'{dest_dir}/{label}'):
                            os.mkdir(f'{dest_dir}/{label}')
                        img.save(f'{dest_dir}/{label}/{"".join(path.name.split(".")[:-1])}.jpg', "JPEG", quality=100)
                    except UnidentifiedImageError:
                        pass


if __name__ == '__main__':

    # load pre-trained w2v model
    # to change
    # fasttext_model = FastText.load('/content/fasttext/fasttext_model_full-1.bin').wv

    w2v_model = Word2Vec.load('../models/w2v_model.bin')
    nlp_model = w2v_model.wv

    dest_imgs_dir = '/Users/salvo/Documents/tesi/datasets/images'
    dest_txts_dir = '/Users/salvo/Documents/tesi/datasets/snippets'

    if not os.path.exists(dest_imgs_dir):
        os.mkdir(dest_imgs_dir)

    if not os.path.exists(dest_txts_dir):
        os.mkdir(dest_txts_dir)

    load_img_samples('/Users/salvo/Documents/tesi/datasets/snippets_2000-47',
                     dest_imgs_dir)

    load_txt_samples('/Users/salvo/Documents/tesi/datasets/snippets_2000_47_score-2_acc-false_rows-2',
                     dest_txts_dir, nlp_model)
