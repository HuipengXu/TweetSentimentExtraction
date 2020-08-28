from allennlp.data.token_indexers import PretrainedTransformerIndexer, TokenIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer, Tokenizer, Token
from allennlp.data.fields import TextField, SpanField
from allennlp.data import DatasetReader
from allennlp.data.instance import Instance

import pandas as pd

from typing import List, Dict, Iterable

@DatasetReader.register("sentiment-df")
class SentimentDfReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None):
        super().__init__(lazy)
        self.tokenizer = tokenizer or PretrainedTransformerTokenizer('bert-base-cased')
        self.token_indexers = token_indexers or {'bert_tokens': PretrainedTransformerIndexer('bert-base-cased')}
        self.max_tokens = max_tokens

    def text_to_instance(self,
                         tokens: List[Token],
                         label: List[Token],
                         span: List[int] = None) -> Instance:
        if self.max_tokens:
            tokens = tokens[:self.max_tokens-2]
        tokens.extend(label)
        text_field = TextField(tokens, self.token_indexers)
        fields = {'text': text_field}
        if span:
            fields['span'] = SpanField(*span, text_field)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            tokens = self.tokenizer.tokenize(row.text)
            str_tokens = [str(token) for token in tokens]
            selected_tokens = self.tokenizer.tokenize(row.selected_text)
            start = str_tokens.index(str(selected_tokens[1]))
            end = len(str_tokens) - str_tokens[::-1].index(str(selected_tokens[-2])) - 1
            span = [start, end]
            sentiment = self.tokenizer.tokenize(row.sentiment)[1:]
            yield self.text_to_instance(tokens, sentiment, span)