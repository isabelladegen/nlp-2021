"""
This script is used to convert the raw data for training and evaluation
"""

from gensim.models.doc2vec import TaggedDocument
from datasets import load_dataset
from gensim.utils import simple_preprocess
import pandas as pd
from src.configurations import Configuration


def load_documents_df(config: dict = {}):
    """
    Load document dataset from huggingface into df

    Returns:
        [numpy dataframe]: slimmed down numpy dataframe
   """
    params = Configuration(**config)

    document_dataset = load_dataset(
        params.dataset_name,
        name=params.document_data_name,
        split=params.document_data_split,
        ignore_verifications=params.data_ignore_verifications,
        keep_in_memory=params.keep_in_memory,
        cache_dir=params.data_cache_dir
    )
    document = pd.DataFrame(data=document_dataset)
    # drop columns we don't need
    document.drop(['title', 'doc_text', 'doc_html_ts', 'doc_html_raw'], axis=1, inplace=True)
    return document


# TODO do something a bit more clever, e.g don't remove numbers
def preprocess_doc(doc: str):
    """
    Preprocessing of the string

    Args:
       doc str: document as string, e.g span

    Returns:
       []: tokenized and processed str
    """
    return simple_preprocess(doc, deacc=True)


def grounding_documents_for_dataframe(df: pd.DataFrame, limit_doc_ids_to: set[str] = None):
    result = []
    if not limit_doc_ids_to:  # no doc id limitation
        for index, row in df.iterrows():
            result.append(GroundingDocument(row))
    else:  # only great grounding doc for those that contain a provide doc id
        limited_df = df.loc[df['doc_id'].isin(limit_doc_ids_to)]
        for index, row in limited_df.iterrows():
            result.append(GroundingDocument(row))
    return result


class GroundingDocument:
    """
    Representing a preprocessed grounding document

    Args:
       doc_row [objec]: ndarray row for one object with doc_id and spans
    """
    id: str  # grounding document id
    raw_docs: {str, str}  # key= span_id, value=span_text
    preprocessed_spans: {str, str}
    tagged_documents: [TaggedDocument]

    def __init__(self, doc_row: object):
        self.id = doc_row['doc_id']
        self.raw_docs = self.__spans_dict(doc_row['spans'])
        self.preprocessed_spans = self.__preprocess_spans()
        self.tagged_documents = self.__create_tagged_documents()

    def __spans_dict(self, spans: [{}]) -> {str, str}:
        """
        Args:
            spans: list of span dictionaries
        Returns:
            result: dictionary with key=sp_id and value = sp_text
        """
        result = {}
        for span in spans:
            result[span['id_sp']] = span['text_sp']
        return result

    def __preprocess_spans(self) -> {str, str}:
        """
        Args:
            none
        Returns:
            {}: dictionary with key=sp_id and value = preprocessed sp_text
        """
        result = {}
        for key, value in self.raw_docs.items():
            result[key] = preprocess_doc(value)
        return result

    def __create_tagged_documents(self):
        """
        Args:
            none
        Returns:
            []: list of TaggedDocuments for the preprocessed spans
        """
        result = []
        for key, value in self.preprocessed_spans.items():
            # !!!hugely important to add the key in a list as otherwise the string will be broken up and more than \
            # one vector will be created per document!!!
            result.append(TaggedDocument(value, [key]))
        return result

    def tagged_document_for(self, span_id: str) -> TaggedDocument:
        return next((doc for doc in self.tagged_documents if doc.tags[0] == span_id), None)

    def original_text_for_sp_id(self, span_id):
        return self.raw_docs[span_id]

    def processed_text_for_sp_id(self, span_id):
        return self.preprocessed_spans[span_id]
