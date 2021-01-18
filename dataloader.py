from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe

class Dataloader:

    def __init__(self, name, length, dimension=50, vocab_size=25000):
        self.TEXT = data.Field(lower=True, fix_length=length, batch_first=True)
        self.LABEL = data.Field(sequential=False,)

        if name == 'SST':
            self.train, self.dev, self.test = data.TabularDataset.splits(
                path='SST-2', train='train.tsv', validation='dev.tsv',
                test='test.tsv', format='tsv', skip_header=True,
                fields=[('text', self.TEXT), ('label', self.LABEL)])
            print("the size of train: {}, dev:{}, test:{}".format(len(self.train.examples), len(self.dev.examples), len(self.test.examples)))

            self.TEXT.build_vocab(self.train, vectors=GloVe(name='6B', dim=dimension), max_size=vocab_size)
            self.LABEL.build_vocab(self.train,)

            print("train.fields:", self.train.fields, self.TEXT.vocab.vectors.shape)

    def split(self, batch_sizes=(8, 32, 32)):

        train_iter, dev_iter, test_iter = data.BucketIterator.splits(
                (self.train, self.dev, self.test), batch_sizes=batch_sizes, sort_key=lambda x: len(x.text), sort_within_batch=True, repeat=False
            )
        train_iter.repeat = False
        test_iter.repeat = False

        return train_iter, dev_iter, test_iter