from copy import deepcopy
from typing import List, Tuple, Dict

from haystack import component, Document
from more_itertools import windowed


@component
class CustomDocumentSplitter:

    def __init__(
        self,
        split_length: int = 200,
        split_overlap: int = 0,
    ):

        self.split_by = "\n"
        if split_length <= 0:
            raise ValueError("split_length must be greater than 0.")
        self.split_length = split_length
        if split_overlap < 0:
            raise ValueError("split_overlap must be greater than or equal to 0.")
        self.split_overlap = split_overlap

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):

        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            raise TypeError("DocumentSplitter expects a List of Documents as input.")

        split_docs = []
        for doc in documents:
            if doc.content is None:
                raise ValueError(
                    f"DocumentSplitter only works with text documents but document.content for document ID {doc.id} is None."
                )
            units = self._split_into_units(doc.content)
            text_splits, splits_pages = self._concatenate_units(units, self.split_length, self.split_overlap)
            metadata = deepcopy(doc.meta)
            metadata["source_id"] = doc.id
            split_docs += self._create_docs_from_splits(
                text_splits=text_splits, splits_pages=splits_pages, meta=metadata
            )
        return {"documents": split_docs}

    def _split_into_units(self, text: str) -> List[str]:
        split_at = "\n"
        units = text.split(split_at)
        # Add the delimiter back to all units except the last one
        for i in range(len(units) - 1):
            units[i] += split_at
        return units

    def _concatenate_units(
        self, elements: List[str], split_length: int, split_overlap: int
    ) -> Tuple[List[str], List[int]]:

        text_splits = []
        splits_pages = []
        cur_page = 1
        segments = windowed(elements, n=split_length, step=split_length - split_overlap)
        for seg in segments:
            current_units = [unit for unit in seg if unit is not None]
            txt = "".join(current_units)
            if len(txt) > 0:
                text_splits.append(txt)
                splits_pages.append(cur_page)
                processed_units = current_units[: split_length - split_overlap]
                if self.split_by == "page":
                    num_page_breaks = len(processed_units)
                else:
                    num_page_breaks = sum(processed_unit.count("\f") for processed_unit in processed_units)
                cur_page += num_page_breaks
        return text_splits, splits_pages

    @staticmethod
    def _create_docs_from_splits(text_splits: List[str], splits_pages: List[int], meta: Dict) -> List[Document]:
        """
        Creates Document objects from text splits enriching them with page number and the metadata of the original document.
        """
        documents: List[Document] = []

        for i, txt in enumerate(text_splits):
            meta = deepcopy(meta)
            doc = Document(content=txt, meta=meta)
            doc.meta["page_number"] = splits_pages[i]
            documents.append(doc)
        return documents
