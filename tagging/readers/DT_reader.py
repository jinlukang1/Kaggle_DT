from typing import Dict, List, Iterator
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer, PretrainedBertIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.token import show_token

import csv


@DatasetReader.register("DT_reader")
class DTDatasetReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, 'r') as conll_file:
            # itertools.groupby is a powerful function that can group
            # successive items in a list by the returned function call.
            # In this case, we're calling it with `is_divider`, which returns
            # True if it's a blank line and False otherwise.
            csv_reader = csv.reader(conll_file)
            head = next(csv_reader)
            for line in csv_reader:
                if len(line) == 4:
                    line.append('0')
                data_id,keyword,location,text,label = line
                data_id = [data_id]
                keyword = keyword.split()
                location = location.split()
                text = text.split()
                if len(keyword) == 0:
                    keyword = ['null']
                if len(location) == 0:
                    location = ['null']
                yield self.text_to_instance(data_id = data_id, 
                                            text = text, 
                                            keyword = keyword, 
                                            location = location,
                                            label = label)

    @overrides
    def text_to_instance(self,
                         data_id: List[str],
                         text: List[str], 
                         keyword: List[str],
                         location: List[str],
                         label: int) -> Instance:
        fields: Dict[str, Field] = {}
        # wrap each token in the file with a token object
        text_field = TextField([Token(w) for w in text], self._token_indexers)
        keyword_field = TextField([Token(w) for w in keyword], self._token_indexers)
        location_field = TextField([Token(w) for w in location], self._token_indexers)
        data_id_field = TextField([Token(w) for w in data_id], self._token_indexers)
        # Instances in AllenNLP are created using Python dictionaries,
        # which map the token key to the Field type
        fields["text"] = text_field
        fields["keyword"] = keyword_field
        fields["location"] = location_field
        fields["label"] = LabelField(label)
        fields["data_id"] = data_id_field

        return Instance(fields)

if __name__ == "__main__":
    reader = DTDatasetReader()
    train_dataset = reader.read('/Users/jinlukang/Desktop/JD/NLP/Disaster_Tweets/data/train.csv')
    print(train_dataset[100])

    