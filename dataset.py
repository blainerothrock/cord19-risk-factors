import json
import os
from torchtext import data
from torchtext.datasets import WikiText2, LanguageModelingDataset

class Cord19(LanguageModelingDataset):
    """
    REQUIREMENTS: needs to have folder structure: cord19_data/cord19/cord19/<train/validation/test>_tokens.txt.
    Unzip corpus.json.zip to get corpus.json and then run json_to_txt.py to create the above folder and files ^
    """

    name = 'cord19'
    dirname = 'cord19'

    @classmethod
    def splits(cls, text_field, root='.data', train='train_tokens.txt',
               validation='validation_tokens.txt', test='test_tokens.txt',
               **kwargs):
        """Create dataset objects for splits of the WikiText-2 dataset.

        This is the most flexible way to use the dataset.

        Arguments:
            text_field: The field that will be used for text data.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose wikitext-2
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'wiki.train.tokens'.
            validation: The filename of the validation data, or None to not
                load the validation set. Default: 'wiki.valid.tokens'.
            test: The filename of the test data, or None to not load the test
                set. Default: 'wiki.test.tokens'.
        """
        return super(Cord19, cls).splits(
            root=root, train=train, validation=validation, test=test,
            text_field=text_field, **kwargs)

    @classmethod
    def iters(cls, batch_size=32, bptt_len=35, device=0, root='cord19_data',
              vectors=None, **kwargs):
        """Create iterator objects for splits of the WikiText-2 dataset.

        This is the simplest way to use the dataset, and assumes common
        defaults for field, vocabulary, and iterator parameters.

        Arguments:
            batch_size: Batch size.
            bptt_len: Length of sequences for backpropagation through time.
            device: Device to create batches on. Use -1 for CPU and None for
                the currently active GPU device.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose wikitext-2
                subdirectory the data files will be stored.
            wv_dir, wv_type, wv_dim: Passed to the Vocab constructor for the
                text field. The word vectors are accessible as
                train.dataset.fields['text'].vocab.vectors.
            Remaining keyword arguments: Passed to the splits method.
        """
        TEXT = data.Field()

        train, val, test = cls.splits(TEXT, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)

        return data.BPTTIterator.splits(
            (train, val, test), batch_size=batch_size, bptt_len=bptt_len,
            device=device)


# test it out
train, val, test = Cord19.iters(batch_size=32, bptt_len=100)

print(train)
print(val)
print(test)
