from typing import Dict
import re
import logging
import csv
import zipfile
import io

from overrides import overrides

from allennlp.common import Params
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
        file_name, member = parse_file_uri(file_path)
        
        if member is None:
            with open(cached_path(file_path), "r") as data_file:
                logger.info("Reading instances from lines in file at: %s", file_path)
                tsvin = csv.reader(data_file, delimiter='\t')
                for row in tsvin:
                    if len(row) != 4:
                        continue
                    label, s1, s2 = row[0], row[1], row[2]
                    yield self.text_to_instance(label, s1, s2)
        else:
            with zipfile.ZipFile(file_name, 'r') as my_zip:
                with my_zip.open(member, "r") as member_file:
                    logger.info("Reading instances from lines in file at: %s", file_path)
                    data_file = io.TextIOWrapper(member_file)
                    tsvin = csv.reader(data_file, delimiter='\t')
                    for row in tsvin:
                        if len(row) != 4:
                            continue
                        label, s1, s2 = row[0], row[1], row[2]
                        yield self.text_to_instance(label, s1, s2)

    @overrides
    def text_to_instance(self, label: str, s1: str, s2: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_s1 = self._tokenizer.tokenize(s1)
        tokenized_s2 = self._tokenizer.tokenize(s2)
        s1_field = TextField(tokenized_s1, self._token_indexers)
        s2_field = TextField(tokenized_s2, self._token_indexers)
        fields = {'label': LabelField(label), 's1': s1_field, 's2': s2_field}

        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'QuoraParaphraseDatasetReader':
        lazy = params.pop('lazy', False)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        params.assert_empty(cls.__name__)
        return cls(lazy=lazy, tokenizer=tokenizer, token_indexers=token_indexers)
