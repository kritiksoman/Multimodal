import docx

from .model_loader import ModelLoader
from pdfminer.high_level import extract_text
import os


class Text(ModelLoader):
    def __init__(self, tasks, model_path):
        super(Text, self).__init__(self, model_path)
        self.doc = {}
        self.file_doc = {}
        self.pdf_path = None
        self.docx_path = None
        self.docx = None
        # self.tasks = ["ner"]

    def _load_pdf(self, pdf_path=None, maxpages=2, page_numbers=None):
        if os.path.exists(pdf_path):
            self.pdf_path = pdf_path
        self.file_doc[self.pdf_path] = [extract_text(pdf_path, maxpages=maxpages, page_numbers=page_numbers)]

    def _load_docx(self, docx_path=None):
        if os.path.exists(docx_path):
            self.docx_path = docx_path
        self.docx = docx.Document(self.docx_path)
        self.file_doc[self.docx_path] = [para.text for para in self.docx.paragraphs]

    def _merge_rec_ner(self, text, prediction):
        merged_prediction = []
        while len(prediction) > 0:
            prediction.sort(key=lambda x: x['start'])
            current = prediction[0]
            current_tag = prediction[0]['entity_group']
            neighbours = list(filter(
                lambda x: (current['start'] == x['end'] or current['end'] == x['start']) and current['entity_group'] ==
                          x['entity_group'], prediction))
            if len(neighbours) >= 1:
                start = min([current] + neighbours, key=lambda x: x['start'])['start']
                end = max([current] + neighbours, key=lambda x: x['end'])['end']
                score = min([current] + neighbours, key=lambda x: x['score'])['score']
                prediction.append({'start': start, 'end': end, 'entity_group': current_tag, 'score': score})
                for neighbour in neighbours:
                    prediction.remove(neighbour)
            else:
                merged_prediction.append(
                    {'beginPosition': current['start'], 'endPosition': current['end'], 'conceptType': current_tag,
                     'score': current['score']})
            prediction.remove(current)
        for entity in merged_prediction:
            entity['text'] = text[entity['beginPosition']:entity['endPosition']]
        return merged_prediction

