import streamlit as st

import spacy
from spacy.tokens import Doc
from spacy.gold import spans_from_biluo_tags
from spacy.gold import iob_to_biluo
from spacy import displacy
from spacy.lang.en import English

from collections import Counter
import pandas as pd
import random
from utils.reader import parse_csv_file

datasets = {
    "Conll2012-train": {
        "location": "/Users/kaushik/Code/experimental/OntoNotes-5.0-NER-BIO/ontov2.train.ner",
        "type": "ner"
    },
    "Conll2012-dev": {
        "location": "/Users/kaushik/Code/experimental/OntoNotes-5.0-NER-BIO/ontov2.development.ner",
        "type": "ner"
    },
    "Redbull-train": {
        "location": "/Users/kaushik/Code/passageai/offline-data/ner/redbull/train.ner",
        "type": "ner"
    },
    "Redbull-dev": {
        "location": "/Users/kaushik/Code/passageai/offline-data/ner/redbull/dev.ner",
        "type": "ner"
    },
    "Tempeval-train": {
        "location": "/Users/kaushik/Code/passageai/offline-data/ner/time/train.ner"
    },
    "Tempeval-dev": {
        "location": "/Users/kaushik/Code/passageai/offline-data/ner/time/dev.ner"
    },
    "ITSM-train": {
        "location": "/Users/kaushik/Code/passageai/offline-data/ner/snow/train.ner"
    },
    "ITSM-dev": {
        "location": "/Users/kaushik/Code/passageai/offline-data/ner/snow/dev.ner"
    },
    "ATIS-train": {
        "location": "/Users/kaushik/Code/passageai/offline-data/ner/atis/train.ner"
    },
    "ATIS-dev": {
        "location": "/Users/kaushik/Code/passageai/offline-data/ner/atis/dev.ner"
    }
}
attrs = ["entity", "value", "start_word", "doc_span"]
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

st.sidebar.title("Pick any NER Dataset")
st.sidebar.markdown(
    """
Pick any NER Dataset to view statistics (num documents) and also view
the entities in table form and also through highlighting. This will hopefully
help us understand the datasets better.
"""
)

@st.cache(allow_output_mutation=True)
def load_model():
    return English()


ner_dataset = st.sidebar.selectbox("Datasets", list(datasets.keys()))
nlp = load_model()


@st.cache(allow_output_mutation=True)
def load_dataset(path, ndocs=None):
    """Returns a list of spacy docs and some stats
    """
    documents = []  # {}
    stats = Counter()
    per_entity_docs = {}

    tokens_and_iob = []
    for row in parse_csv_file(path, delimiter='\t'):
        if row:
            if len(row) != 4:
                continue
            tokens_and_iob.append((row[0], row[3]))
        else:
            if len(tokens_and_iob) == 0:
                continue
            tokens, iob = zip(*tokens_and_iob)
            num_initial_entities = len(list(filter(
                lambda x: x.startswith("B-") or x.startswith("B_"), iob)))
            if num_initial_entities == 0:
                tokens_and_iob = []
                stats["empty_docs"] += 1
                continue
            doc = Doc(nlp.vocab, tokens)
            biluo_tags = iob_to_biluo(iob)
            doc.ents = spans_from_biluo_tags(doc, biluo_tags)
            for ent in doc.ents:
                if ent.label_ not in per_entity_docs:
                    per_entity_docs[ent.label_] = []
                per_entity_docs[ent.label_].append(doc)
            num_predicted_entities = len(doc.ents)
            if num_predicted_entities != num_initial_entities:
                stats["unaligned_count"] += 1
                tokens_and_iob = []
                continue
            documents.append(doc)
            tokens_and_iob = []
            stats["doc_count"] += 1
            if ndocs and stats["doc_count"] == ndocs:
                break
    return documents, per_entity_docs, stats


with st.spinner('Loading Dataset {}'.format(ner_dataset)):
    docs, per_entity_docs, stats = load_dataset(
        datasets[ner_dataset]["location"])
st.success('Loaded Datset {}!'.format(ner_dataset))

st.title("Understanding the dataset: {}".format(ner_dataset))
st.write("The length of the data is: ", stats["doc_count"])
st.write("The length of the misaligned entities is: ", stats["unaligned_count"])
st.write("The length of data with no entities is: ", stats["empty_docs"])


def display_docs_in_table(docs, ndocs=1000, filter_entity=None):
    entities = []
    sample_number = min(len(docs), ndocs)
    for doc in random.sample(docs, sample_number):
        doc_text = doc.text
        for ent in doc.ents:
            if not filter_entity or (filter_entity and ent.label_ == filter_entity):
                min_display_char = max(0, ent.start_char - 10)
                max_display_char = min(len(doc_text), ent.end_char + 10)
                entities.append([
                    ent.label_,
                    ent.text,
                    ent.start,
                    doc_text[min_display_char:max_display_char]
                ])
    df = pd.DataFrame(entities, columns=attrs)
    st.dataframe(df)


def visualize_docs(docs, ndocs=20):
    sample_number = min(len(docs), ndocs)
    for i, doc in enumerate(random.sample(docs, sample_number)):
        html = displacy.render(doc, style="ent")
        # Newlines seem to mess with the rendering
        html = html.replace("\n", " ")
        st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)

if st.checkbox('Sample random 1000 documents'):
    display_docs_in_table(docs)

if st.checkbox('Visualize random 20 documents'):
    visualize_docs(docs)

st.title("Diving Deeper into the dataset: {}".format(ner_dataset))
for entity, entity_docs in per_entity_docs.items():
    if st.checkbox(entity):
        st.write("\tNumber of docs with entity {}: ".format(entity), len(entity_docs))
        if st.checkbox("\tSample random 100 {}".format(entity)):
            display_docs_in_table(entity_docs, filter_entity=entity)
        if st.checkbox('\tVisualize random 20 {}'.format(entity)):
            visualize_docs(entity_docs)
