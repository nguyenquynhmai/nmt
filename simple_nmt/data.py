# For data loading
from torchtext import data, datasets
import numpy as np
import torch

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

import spacy 
spacy_de = spacy.load('fr_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
LOWER = True

class Corpus:
    def __init__(self, lang1, lang2):
        
        self.SRC = data.Field(tokenize=tokenize_de, batch_first=True, lower=LOWER, include_lengths=True,
                unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=None, eos_token=EOS_TOKEN)

        self.TRG = data.Field(tokenize=tokenize_en, batch_first=True, lower=LOWER, include_lengths=True,
                unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=None, eos_token=EOS_TOKEN)


        MAX_LEN = 25 # Note: we filter out a lot of sentences for speed
        self.train_data, self.valid_data, self.test_data = datasets.IWSLT.splits(
                    exts=('.{}'.format(lang1), '.{}'.format(lang2)), fields=(self.SRC, self.TRG),
                    filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN)


        MIN_FREQ = 5 # Note: we limite the vocabulary to frequent words for speed
        self.SRC.build_vocab(self.train_data.src, min_freq=MIN_FREQ)
        self.TRG.build_vocab(self.train_data.trg, min_freq=MIN_FREQ)

        self.pad_index = self.TRG.vocab.stoi[PAD_TOKEN]


def print_data_info(train_data, valid_data, test_data, src_field, trg_field):
    """ This prints some useful stuff about our data sets. """

    print("Data set sizes (number of sentence pairs):")
    print('train', len(train_data))
    print('valid', len(valid_data))
    print('test', len(test_data), "\n")

    print("First training example:")
    print("src:", " ".join(vars(train_data[0])['src']))
    print("trg:", " ".join(vars(train_data[0])['trg']), "\n")

    print("Most common words (src):")
    print("\n".join(["%10s %10d" % x for x in src_field.vocab.freqs.most_common(10)]), "\n")
    print("Most common words (trg):")
    print("\n".join(["%10s %10d" % x for x in trg_field.vocab.freqs.most_common(10)]), "\n")

    print("First 10 words (src):")
    print("\n".join(
        '%02d %s' % (i, t) for i, t in enumerate(src_field.vocab.itos[:10])), "\n")
    print("First 10 words (trg):")
    print("\n".join(
        '%02d %s' % (i, t) for i, t in enumerate(trg_field.vocab.itos[:10])), "\n")

    print("Number of German words (types):", len(src_field.vocab))
    print("Number of English words (types):", len(trg_field.vocab), "\n")
    

if __name__ == "__main__":
    corpus =  Corpus('de', 'en')
    print_data_info(corpus.train_data, corpus.valid_data, corpus.test_data, corpus.SRC, corpus.TRG)