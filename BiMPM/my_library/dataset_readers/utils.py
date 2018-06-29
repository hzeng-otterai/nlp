
import torch
from torch.autograd import Variable

from collections import Counter

from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe, Vectors, Vocab

def tokenizer(text): # create a tokenizer function
    return text.split(' ')
    
    
rep_dict = {
    "``": "\"",
    "''": "\"",
    "-LRB-": "(",
    "-RRB-": ")",
    "-LSB-": "[",
    "-RSB-": "]",
}

def preprocessor(token_list):
    result = [rep_dict.get(x, x) for x in token_list]
    return result


class Paraphrase():
    def __init__(self, args):
        self.args = args

    def build_char_vocab(self):
        char_counter = Counter()
        for word, count in self.TEXT.vocab.freqs.items():
            for ch in word:
                char_counter[ch] += count
        
        self.char_vocab = Vocab(char_counter, specials=['<unk>', '<pad>', '<w>', '</w>'])

    def characterize_word(self, word_id, max_word_len):
        if word_id in range(4): # for <unk>, <pad>, <s> and </s>
            w_char = [0]
        else:
            w_char = [2]
            w = self.TEXT.vocab.itos[word_id]
            w_char += [self.char_vocab.stoi[ch] for ch in w[:max_word_len]]
            w_char += [3]
            
        word_len = len(w_char)
        w_char.extend([1] * (max_word_len - word_len))

        return w_char, word_len

    def characterize(self, batch, length):
        """
        :param batch: Pytorch Variable with shape (batch, seq_len)
        :param length: Pytorch Variable with shape (batch)
        :return: Pytorch Variable with shape (batch, seq_len, max_word_len)
        """
        batch = batch.tolist()
        length = length.tolist()
        assert len(batch) == len(length) > 0
        actual_max_word_len = 0
        for idx, sent in enumerate(batch):
            sent_len = length[idx]
            assert 2 <= sent_len <= len(sent)
            # get max length of word, excluding <s> and </s>
            actual_max_word_len = max(actual_max_word_len, max([len(self.TEXT.vocab.itos[w_id]) for w_id in sent[1:sent_len-1]]))

        if self.args.max_word_len <= 0:
            max_word_len = actual_max_word_len
        else:
            max_word_len = min(actual_max_word_len, self.args.max_word_len)

        assert(max_word_len > 0)
        max_word_len += 2 # adding <w> and </w>

        char_result = []
        char_len = []
        for idx, sent in enumerate(batch):
            sent_len = length[idx]
            padding_len = len(sent) - sent_len
            current_char_result = []
            current_char_len = []
            for w_id in sent[:sent_len]:
                cr, cl = self.characterize_word(w_id, max_word_len)
                current_char_result.append(cr)
                current_char_len.append(cl)

            # padding the sentence with [<unk> <pad> <pad> ...] 
            # since pack_padded_sequence doesn't accept zero length input
            for i in range(padding_len):
                current_char_result.append([0] + [1] * (max_word_len - 1))
            current_char_len.extend([1] * padding_len)

            char_result.append(current_char_result)
            char_len.append(current_char_len)

        return Variable(torch.LongTensor(char_result)), Variable(torch.LongTensor(char_len))

    def get_features(self, batch):
        if self.args.data_type == 'SNLI':
            s1, s2 = 'premise', 'hypothesis'
        else:
            s1, s2 = 'q1', 'q2'
            
        s1 = getattr(batch, s1)
        s2 = getattr(batch, s2)

        s1, s1_len = s1[0], s1[1]
        s2, s2_len = s2[0], s2[1]

        max_p_len = s1_len.max().item()
        max_h_len = s2_len.max().item()

        s1 = s1[:, :max_p_len]
        s2 = s2[:, :max_h_len]

        kwargs = {}
        kwargs.update({'p': s1, 'p_len': s1_len})
        kwargs.update({'h': s2, 'h_len': s2_len})

        char_p, char_p_len = self.characterize(s1, s1_len)
        char_h, char_h_len = self.characterize(s2, s2_len)

        if self.args.gpu > -1:
            char_p = char_p.cuda(self.args.gpu)
            char_h = char_h.cuda(self.args.gpu)
            char_p_len = char_p_len.cuda(self.args.gpu)
            char_h_len = char_h_len.cuda(self.args.gpu)

        kwargs.update({'char_p': char_p, 'char_p_len': char_p_len, 'char_h': char_h, 'char_h_len': char_h_len})

        return kwargs

class SNLI(Paraphrase):
    def __init__(self, args):
    
        super().__init__(args)
        
        fix_length = args.max_sent_len if args.max_sent_len >=0 else None
        
        self.TEXT = data.Field(batch_first=True, init_token="<s>", eos_token="</s>", preprocessing=preprocessor, fix_length=fix_length, include_lengths=True, tokenize="spacy")
        self.LABEL = data.Field(sequential=False, unk_token=None)

        self.train, self.dev, self.test = datasets.SNLI.splits(self.TEXT, self.LABEL)

        self.TEXT.build_vocab(self.train, self.dev, self.test, vectors=GloVe(name='840B', dim=300))
        self.build_char_vocab()
        self.LABEL.build_vocab(self.train)

        self.train_iter, self.dev_iter, self.test_iter = \
            data.BucketIterator.splits((self.train, self.dev, self.test),
                                       batch_sizes=[args.batch_size] * 3,
                                       device=torch.device('cuda', args.gpu) if args.gpu >= 0 else torch.device('cpu'),
                                       repeat=False)


class Quora(Paraphrase):
    def __init__(self, args):
        super().__init__(args)

        fix_length = args.max_sent_len if args.max_sent_len >=0 else None
        
        self.RAW = data.RawField()
        self.TEXT = data.Field(batch_first=True, init_token="<s>", eos_token="</s>", preprocessing=preprocessor, fix_length=fix_length, include_lengths=True, tokenize=tokenizer)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        self.train, self.dev, self.test = data.TabularDataset.splits(
            path='.data/quora',
            train='train.tsv',
            validation='dev.tsv',
            test='test.tsv',
            format='tsv',
            fields=[('label', self.LABEL),
                    ('q1', self.TEXT),
                    ('q2', self.TEXT),
                    ('id', self.RAW)])

        self.TEXT.build_vocab(self.train, self.dev, self.test, vectors=GloVe(name='840B', dim=300))
        self.build_char_vocab()
        self.LABEL.build_vocab(self.train)

        sort_key = lambda x: data.interleave_keys(len(x.q1), len(x.q2))

        self.train_iter, self.dev_iter, self.test_iter = \
            data.BucketIterator.splits((self.train, self.dev, self.test),
                                       batch_sizes=[args.batch_size] * 3,
                                       device=torch.device('cuda', args.gpu) if args.gpu >= 0 else torch.device('cpu'),
                                       sort_key=sort_key,
                                       repeat=False)

