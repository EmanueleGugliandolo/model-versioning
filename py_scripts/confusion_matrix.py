# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:55:10 2022

@author: 08500217
"""
import json
import spacy
from spacy.training import offsets_to_biluo_tags
from spacy.tokens import DocBin
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from argparse import ArgumentParser
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def get_cleaned_label(label: str):
    if "-" in label:
        return label.split("-")[1]
    else:
        return label
    
def create_total_target_vector(docs):
    target_vector = []
    for doc in docs:
        new = nlp.make_doc(doc["data"])
        entities = [e for e in doc["label"] if e != "-"]
        bilou_entities = offsets_to_biluo_tags(new, entities)
        final = []
        #print(bilou_entities)  #############
        for item in bilou_entities:
            final.append(get_cleaned_label(item))
        target_vector.extend(final)
    return target_vector

def create_prediction_vector(text):
    return [get_cleaned_label(prediction) for prediction in get_all_ner_predictions(text)]

def create_total_prediction_vector(docs: list):
    prediction_vector = []
    for doc in docs:
        prediction_vector.extend(create_prediction_vector(doc["data"]))
    return prediction_vector

def get_all_ner_predictions(text):
    doc = nlp(text)
    entities = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
    bilou_entities = offsets_to_biluo_tags(doc, entities)
    return bilou_entities

def get_model_labels():
    labels = list(nlp.get_pipe("ner").labels)
    labels.append("O")
    return sorted(labels)

def get_dataset_labels():
    return sorted(set(create_total_target_vector(docs)))

def generate_confusion_matrix(docs): 
    classes = sorted(set(create_total_target_vector(docs)))
    y_true = create_total_target_vector(docs)
    y_pred = create_total_prediction_vector(docs)
    #print (y_true)
    #print (y_pred)
    return confusion_matrix(y_true, y_pred, labels=classes)

def plot_pretty_confusion_matrix(cm, classes, output_path, image, csv, norm, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `norm=True`.
    """
   
    title = 'Confusion Matrix, for SpaCy NER'
    
    # Normalization
    if norm == True:
        cm = normalize(cm, axis=0, norm='l1')
        cm_format=".3f"
        vmax = 0.05
    else:
        cm_format=".0f"
        vmax = 10

    cmdf = pd.DataFrame(cm, columns=classes)
    cmdf.index = classes
    
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(30, 20))
        ax = sns.heatmap(cmdf, cmap="Blues", vmax=vmax, fmt=cm_format, square=True, annot=True)
        ax.set_title(title)
            
    if csv == True:
        cmdf.to_csv(output_path+"/confusion_matrix.csv")
    if image == True:
        plt.savefig(output_path+"/confusion_matrix.png")
    
    plt.show()
    return cm, ax, plt

def create_json_from_spacy(nlp, spacy_data):
    doc_bin = DocBin().from_disk(spacy_data)
    docs = []
    for doc in doc_bin.get_docs(nlp.vocab):
        spans = [ [ent.start_char, ent.end_char, ent.label_] for ent in doc.ents]
        docs.append({"data": doc.text, "label": spans})
    return docs


if __name__ == "__main__":
    
    parser = ArgumentParser(description="Create confusion matrix for a given model and test data")

    parser.add_argument(
        "--model", type=str, required=True, help="Path to spacy model (Required)"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to test data in .spacy or .json format (Required)"
    )
    parser.add_argument(
        "--output_path", type=str, default = ".", help="default to local dir"
    )
    parser.add_argument(
        "--no_normalize", default=True, action="store_false", help="Don't normalize columns"
    )
    parser.add_argument(
        "--no_image", default=True, action="store_false", help="Don't save file as image"
    )
    parser.add_argument(
        "--no_csv", default=True, action="store_false", help="Don't save file as csv"
    )

    args = parser.parse_args()
    
    nlp = spacy.load(args.model)
    
    if args.data.endswith(".spacy"):
        docs = create_json_from_spacy(nlp, args.data)
    else:
        with open(args.data,"r") as f:
            docs = json.load(f)
            
    plot_pretty_confusion_matrix(generate_confusion_matrix(docs), get_dataset_labels(), args.output_path, image = args.no_image, csv = args.no_csv, norm = args.no_normalize)