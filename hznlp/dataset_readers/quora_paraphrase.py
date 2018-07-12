from typing import Dict
import re
import logging
import csv
import zipfile
import io

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def parse_file_uri(uri: str):
    match = re.fullmatch(r'\((.*)\)#(.*)', uri)      # pylint: disable=anomalous-backslash-in-string
    if match and len(match.groups()) == 2:
        return match.groups()[0], match.groups()[1]
    else:
        return uri, None

@DatasetReader.register("quora_paraphrase")
class QuoraParaphraseDatasetReader(DatasetReader):
    """
    Reads a Quora paraphrase data

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the title and abstrct into words or other kinds of tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer(JustSpacesWordSplitter())
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        logger.info("Reading instances from lines in file at: %s", file_path)
        file_name, member = parse_file_uri(file_path)
        
        if member is None:
            with open(cached_path(file_path), "r") as data_file:
                tsvin = csv.reader(data_file, delimiter='\t')
                for row in tsvin:
                    if len(row) != 4:
                        continue
                    label, s1, s2 = row[0], row[1], row[2]
                    yield self.text_to_instance(s1, s2, label)
        else:
            with zipfile.ZipFile(cached_path(file_name), 'r') as zip_file:
                with zip_file.open(member, "r") as member_file:
                    data_file = io.TextIOWrapper(member_file)
                    tsvin = csv.reader(data_file, delimiter='\t')
                    for row in tsvin:
                        if len(row) != 4:
                            continue
                        label, s1, s2 = row[0], row[1], row[2]
                        yield self.text_to_instance(s1, s2, label)

    @overrides
    def text_to_instance(self, s1: str, s2: str, label: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_s1 = self._tokenizer.tokenize(s1)
        tokenized_s2 = self._tokenizer.tokenize(s2)
        s1_field = TextField(tokenized_s1, self._token_indexers)
        s2_field = TextField(tokenized_s2, self._token_indexers)
        fields = {'s1': s1_field, 's2': s2_field}
        if label is not None:
            fields['label'] = LabelField(label)

        return Instance(fields)

