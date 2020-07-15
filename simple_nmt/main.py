import numpy as np
import torch 
import math, copy, time
import matplotlib.pyplot as plt
from model import *
from torchtext import data
import argparse
import shutil

from data import *

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0') # or set to 'cpu'


# we define a function from hyperparameters to a full model

def make_model(src_vocab, tgt_vocab, emb_size=256, hidden_size=512, num_layers=1, dropout=0.1):
    "Helper: Construct a model from hyperparameters."

    attention = BahdanauAttention(hidden_size)

    model = EncoderDecoder(
        Encoder(emb_size, hidden_size, num_layers=num_layers, dropout=dropout),
        Decoder(emb_size, hidden_size, attention, num_layers=num_layers, dropout=dropout),
        nn.Embedding(src_vocab, emb_size),
        nn.Embedding(tgt_vocab, emb_size),
        Generator(hidden_size, tgt_vocab)
    )
    return model.cuda() if USE_CUDA else model


class Batch:
    """Object for holding a batch of data with mask during training
    Input is a batch from a torch text iterator.
    """
    def __init__(self, src, trg, pad_index=0):
        src, src_lengths = src

        self.src = src
        self.src_lengths = src_lengths
        self.src_mask = (src != pad_index).unsqueeze(-2)
        self.nseqs = src.size(0)

        self.trg = None
        self.trg_y = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None

        if trg is not None:
            trg, trg_lengths = trg
            self.trg = trg[:, : -1]
            self.trg_lengths = trg_lengths
            self.trg_y = trg[:, 1:]
            self.trg_mask = (self.trg_y != pad_index)
            self.ntokens = (self.trg_y != pad_index).data.sum().item()

        if USE_CUDA:
            self.src = self.src.cuda()
            self.src_mask = self.src_mask.cuda()

            if trg is not None:
                self.trg = self.trg.cuda()
                self.trg_y = self.trg_y.cuda()
                self.trg_mask = self.trg_mask.cuda()


def run_epoch(data_iter, model, loss_compute, print_every=50):
    """Standard Training and Logging Function"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    print_tokens = 0

    for i, batch in enumerate(data_iter, 1):

        out, _, pre_output = model.forward(batch.src, batch.trg,
                                           batch.src_mask, batch.trg_mask,
                                           batch.src_lengths, batch.trg_lengths)

        loss = loss_compute(pre_output, batch.trg_y, batch.nseqs)
        total_loss += loss
        total_tokens += batch.ntokens
        print_tokens += batch.ntokens

        if model.training and i % print_every == 0:
            elapsed = time.time() - start
            print('| {:5d} step | training loss: {:.6f} | tokens per second: {:.6f}'.format( i, loss/batch.nseqs, print_tokens/elapsed))
            # print('Epoch Step: %d Loss: %f Tokens per sec: %f' %(i, loss/batch.nseqs, print_tokens/elapsed))
            print("-"*80)
            start = time.time()
            print_tokens = 0

    return total_loss/float(total_tokens)


def data_gen(num_words=11, batch_size=16, num_batches=100, length=10, pad_index=0, sos_index=1):
    """Generate random data for a src-tgt copy task."""
    for i in range(num_batches):
        data = torch.from_numpy(np.random.randint(1, num_words, size=(batch_size, length)))
        data[:, 0] = sos_index
        data = data.cuda() if USE_CUDA else data
        src = data[:, 1:]
        trg = data
        src_lengths = [length-1] * batch_size
        trg_lengths = [length] * batch_size

        yield Batch((src, src_lengths), (trg, trg_lengths), pad_index=pad_index)


class SimpleLossCompute:
    """A simple loss compute and train function"""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
        loss = loss / norm 
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

        return loss.data.item() * norm 



# Print examples
def greedy_decode(model, src, src_mask, src_lengths, max_len=100, sos_index=1, eos_index=None):
    """Greedily decode a sentence."""

    with torch.no_grad():
        encoder_hidden, encoder_final = model.encode(src, src_mask, src_lengths)
        prev_y = torch.ones(1, 1).fill_(sos_index).type_as(src)

        trg_mask = torch.ones_like(prev_y)

    output = []
    attention_scores = []
    hidden = None

    for i in range(max_len):
        with torch.no_grad():
            out, hidden, pre_output = model.decode(encoder_hidden, encoder_final, src_mask, prev_y, trg_mask, hidden)

            # we predict from the pre_output layer which is
            # a combination of Decoder state, prev emb, and context
            prob = model.generator(pre_output[:, -1])

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data.item()
        output.append(next_word)
        prev_y = torch.ones(1, 1).type_as(src).fill_(next_word)
        attention_scores.append(model.decoder.attention.alphas.cpu().numpy())

    output = np.array(output)

    # cut off everything start from </s>
    if eos_index is not None:
        first_eos = np.where(output==eos_index)[0]
        if len(first_eos) > 0:
            output = output[:first_eos[0]]
    return output, np.concatenate(attention_scores, axis=1)

def lookup_words(x, vocab=None):
    if vocab is not None:
        x = [vocab.itos[i] for i in x]

    return [str(t) for t in x]


def print_examples(example_iter, model, n=2, max_len=100,
                    sos_index=1,
                    src_eos_index=None,
                    trg_eos_index=None,
                    src_vocab=None, trg_vocab=None):
    model.eval()
    count = 0
    
    if src_vocab is not None and trg_vocab is not None:
        src_eos_index = src_vocab.stoi[EOS_TOKEN]
        trg_sos_index = trg_vocab.stoi[SOS_TOKEN]
        trg_eos_index = trg_vocab.stoi[EOS_TOKEN]
    else:
        src_eos_index = None
        trg_sos_index = 1
        trg_eos_index = None
        
    for i, batch in enumerate(example_iter):
      
        src = batch.src.cpu().numpy()[0, :]
        trg = batch.trg_y.cpu().numpy()[0, :]

        # remove </s> (if it is there)
        src = src[:-1] if src[-1] == src_eos_index else src
        trg = trg[:-1] if trg[-1] == trg_eos_index else trg      
      
        result, _ = greedy_decode(
          model, batch.src, batch.src_mask, batch.src_lengths,
          max_len=max_len, sos_index=trg_sos_index, eos_index=trg_eos_index)
        print("Example #%d" % (i+1))
        print("Src : ", " ".join(lookup_words(src, vocab=src_vocab)))
        print("Trg : ", " ".join(lookup_words(trg, vocab=trg_vocab)))
        print("Pred: ", " ".join(lookup_words(result, vocab=trg_vocab)))
        print()
        
        count += 1
        if count == n:
            break


def train_copy_task():
    """Train the simple copy task."""
    num_words = 11
    criterion = nn.NLLLoss(reduction="sum", ignore_index=0)
    model = make_model(num_words, num_words, emb_size=32, hidden_size=64)
    optim = torch.optim.Adam(model.parameters(), lr=0.0003)
    eval_data = list(data_gen(num_words=num_words, batch_size=1, num_batches=100))
 
    dev_perplexities = []
    
    if USE_CUDA:
        model.cuda()

    for epoch in range(10):
        
        print("Epoch %d" % epoch)

        # train
        model.train()
        data = data_gen(num_words=num_words, batch_size=32, num_batches=100)
        run_epoch(data, model,
                  SimpleLossCompute(model.generator, criterion, optim))

        # evaluate
        model.eval()
        with torch.no_grad(): 
            perplexity = run_epoch(eval_data, model,
                                   SimpleLossCompute(model.generator, criterion, None))
            print("Evaluation perplexity: %f" % perplexity)
            dev_perplexities.append(perplexity)
            print_examples(eval_data, model, n=2, max_len=9)
        
    return dev_perplexities

# train the copy task
# dev_perplexities = train_copy_task()

def plot_perplexity(perplexities):
    """plot perplexities"""
    plt.title("Perplexity per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.plot(perplexities)
    

def rebatch(pad_idx, batch):
    """Wrap torchtext batch into our own Batch class for pre-processing"""
    return Batch(batch.src, batch.trg, pad_idx)


def train(start_epochs, valid_loss_min_input, corpus, model, optimizer, criterion, args):
    """Train a model with GPU"""
    train_iter = data.BucketIterator(corpus.train_data, batch_size=args.batch_size,
                                    train=True, sort_within_batch=True,
                                    sort_key=lambda x: (len(x.src), len(x.trg)), repeat=False,
                                    device=DEVICE)

    valid_iter = data.BucketIterator(corpus.valid_data, batch_size=1,
                                    sort=False, repeat=False, device=DEVICE)

    PAD_INDEX = corpus.pad_index

    if USE_CUDA:
        model.cuda()

    valid_loss_min = valid_loss_min_input
    

    for epoch in range(start_epochs, args.epochs+1):
        print("Epoch", epoch)
        model.train()
        train_loss = run_epoch((rebatch(PAD_INDEX, b) for b in train_iter),
                                        model,
                                        SimpleLossCompute(model.generator, criterion, optimizer),
                                        print_every=args.print_every)

        ######################
        # validate the model #
        ######################
        model.eval()

        with torch.no_grad():
            valid_loss = run_epoch((rebatch(PAD_INDEX, b) for b in valid_iter),
                                model,
                                SimpleLossCompute(model.generator, criterion, None))
            print('Epoch: {:3d} | train loss: {:5.2f} | valid loss: {:5.2f} | valid ppl: {:.6f}'.format(epoch, train_loss, valid_loss, math.exp(valid_loss)))
            print("="*80)

            checkpoint = {
                'epoch': epoch + 1,
                'valid_loss_min': valid_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            # save checkpoint
            save_ckp(checkpoint, False, args.ckp_path, args.best_model_path)
        
            ## TODO: save the model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
                # save checkpoint as best model
                save_ckp(checkpoint, True, args.ckp_path, args.best_model_path)
                valid_loss_min = valid_loss

            print_examples((rebatch(PAD_INDEX, x) for x in valid_iter), model,
                            n=3, src_vocab=corpus.SRC.vocab, trg_vocab=corpus.TRG.vocab)
    return model

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Training seq2seq model")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--src_lang", type=str, default="de", help="source language")
    parser.add_argument("--trg_lang", type=str, default="en", help="target language")
    parser.add_argument("--print_every", type=int, default=100, help="Print every")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--nlayers", type=int, default=2, help="Number of layers")
    parser.add_argument("--emsize", type=int, default=256)
    parser.add_argument("--nhid", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--ckp_path", type=str, default="./checkpoints/current_ckp.pt", help="save path")
    parser.add_argument("--best_model_path", type=str, default="./checkpoints/best_model/best_model.pt")
    
    args = parser.parse_args()

    corpus = Corpus(args.src_lang, args.trg_lang)
    model = make_model(len(corpus.SRC.vocab), len(corpus.TRG.vocab),
                        emb_size=args.emsize, hidden_size=args.nhid,
                        num_layers=args.nlayers, dropout=args.dropout)
    
    # Optionally add label smoothing, see the annotated transformer
    criterion = nn.NLLLoss(reduction="sum", ignore_index=corpus.pad_index)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    if args.resume: 
        model, optimizer, start_epoch, valid_loss_min = load_ckp(args.best_model_path, model, optimizer)
    else:
        valid_loss_min = np.Inf,
        start_epoch = 0
    

    model = train(start_epoch, valid_loss_min, corpus, model, optimizer, criterion, args)