import spacy
from argparse import ArgumentParser
from spacy.tokens import DocBin
from spacy.training import offsets_to_biluo_tags
from spacy.symbols import ORTH, LOWER

import os
import glob
import json
import re
import pandas as pd
import logging
import random

# SETUP LOGGER
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


SPECIAL_CASES = [
    "CnC",
    "C&C",
    "LAPSUS$",
    "Lapsus$",
    "DomaiN",
    "UrL",
    "FilenamE",
    "AsN",
    "DDoS",
    "EDoS",
    "RATs",
    "C2s",
    "RaaS"
]

# Prints a dataframe with the count of the entities in train, valid, test documents
def make_df_count(train, valid, test, save = False):
    
    all_labels = []
    for docs in [train,valid,test]:
        data=[]
        for doc in docs:
            data.append(doc)
        list_of_labels = [j[2] for i in data for j in i['label']]
        all_labels.append(list_of_labels)
        
    df_count = pd.DataFrame(all_labels)
    df_count = df_count.transpose()
    
    df_count = df_count.apply(pd.Series.value_counts)
    df_count.columns = ['Train','Valid','Test']
    df_count.loc["TOTAL"]=df_count.sum(axis=0)
    df_perc = df_count.div(df_count.sum(axis=1), axis=0)
    
    print(df_count)
    print(df_perc)
    if save == 'save':
        df_count.to_csv(index=False)
    return 

def read_json_files(json_filepath):
    """
    read content of json files present in "json_filepath"
    Output: list of dictionary
    """
    paths = glob.glob(json_filepath)
    paths.sort()
    data = []
    for filename in paths:
        data.extend(json.load(open(os.path.join(os.getcwd(), filename))))
    return data


def modify_entities(tagged_document, ents_to_remove=None, ents_to_rename=None):
    """
    Input: dict[
            'text':document,
            ...,
            'label':[
                    [start, end, label],
                    ...,
                    [start, end, label]
                ]
            ]
    Output: returns modified document
    """
    ### REMOVE ENTS
    if ents_to_remove is not None:
        tagged_document['label'] = [label for label in tagged_document['label'] if label[2] not in ents_to_remove]
    ### MODIFY ENTS
    if ents_to_rename is not None:
        ents_to_rename = json.loads(ents_to_rename)
        for element in tagged_document['label']:
            if element[2] in ents_to_rename.keys():
                element[2] = ents_to_rename[element[2]]
    return tagged_document


def json_to_spacy_binary(json_list, filepath, special_cases=None, ents_to_remove=None, ents_to_rename=None, aug_perc=None):
    """
    inputs: list of dict[tagged_document], output filepath, special cases for tokenizer,  list of ents to remove, dict of ents to replace
    """
    number_of_ignored_labels = 0
    number_of_labels = 0
    db = DocBin()
    nlp = spacy.blank("en")
    if special_cases is not None:
        for case in special_cases:
            nlp.tokenizer.add_special_case(case+".", [{ORTH: case}, {ORTH: "."}])
    nlp.tokenizer.add_special_case("'DomaiN'",  [{ORTH: "'"}, {ORTH: "DomaiN"}, {ORTH: "'"}])
    for entry in json_list:
        # modify entities
        if ents_to_remove or ents_to_rename is not None:
            entry = modify_entities(entry, ents_to_remove, ents_to_rename)
        # convert json to spacy span format
        doc = nlp.make_doc(entry['data']) 
        ents = []
        for start, end, label in entry["label"]:
            number_of_labels = number_of_labels+1
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                logger.debug("--DocID: {}  --label: {}  --Text: {}".format(entry['doc_id'], label, entry['data'][start-2:end+2]))
                number_of_ignored_labels = number_of_ignored_labels+1
            else:
                ents.append(span)
        doc.ents = ents 
        db.add(doc)
        
    # Augment
    if aug_perc is not None:
        import nlpaug.augmenter.char as nac
        for entry in json_list[0:int(aug_perc*len(json_list))]:
            # modify entities
            if ents_to_remove or ents_to_rename is not None:
                entry = modify_entities(entry, ents_to_remove, ents_to_rename)
            # augment doc
            text = entry['data']
            aug = nac.KeyboardAug()
            aug_text = aug.augment(text)

            # convert json to spacy span format
            doc = nlp.make_doc(aug_text) 
            ents = []
            for start, end, label in entry["label"]:
                number_of_labels = number_of_labels+1
                span = doc.char_span(start, end, label=label, alignment_mode="contract")
                if span is None:
                    logger.debug("--DocID: {}  --label: {}  --Text: {}".format(entry['doc_id'], label, entry['data'][start-2:end+2]))
                    number_of_ignored_labels = number_of_ignored_labels+1
                else:
                    ents.append(span)
            doc.ents = ents 
            db.add(doc)
    
    # saving
    db.to_disk(filepath)
    logger.debug("---ignored labels: {}   ---total labels: {}  ---number of docs: {}".format(number_of_ignored_labels, number_of_labels, len(json_list)))


def json_to_bio(json_list, special_cases=None, spacy_model=None, text_field="tokens", tags_field="labels", ents_to_remove=None, ents_to_rename=None):
    """
    convert already tagged text from span format to BILUO or IOB format
    Inputs: json list from read_json_files function, special cases for tokenizer, spacy_model, text_field, tags_field, list of entities to remove, dict of entities to rename
    Output: pandas dataframe
        text_field: tokenized documents 
        tags_field: list of ner_tags
    """
    if spacy_model is not None:
        nlp = spacy.load(spacy_model, disable = ['ner','lemmatizer', 'attribute_ruler', 'tagger'])
    else:
        nlp = spacy.blank("en")
        if special_cases is not None:
            for case in special_cases:
                nlp.tokenizer.add_special_case(case+".", [{ORTH: case}, {ORTH: "."}])
        docs = []
        ner_tags = []
        count = 0
        for entry in json_list:
            # modify entities
            if ents_to_remove or ents_to_rename is not None:
                entry = modify_entities(entry, ents_to_remove, ents_to_rename)
            # convert json to bio format
            doc = nlp(entry['data'])
            tags = offsets_to_biluo_tags(doc, entry['label'])
            docs.append([token.text for token in doc])
            ner_tags.append(tags)
            count = count+1

        number_of_labels = 0
        number_of_ignored_labels = 0
        new_ner = []
        for tags in ner_tags:
            new_tags = []
            for tag in tags:
                if tag!='O':
                    number_of_labels = number_of_labels+1
                tag = re.sub('L-','I-',tag)
                tag = re.sub('U-', 'B-',tag)
                if tag=='-':
                    number_of_ignored_labels = number_of_ignored_labels+1
                    tag = re.sub('-', 'O',tag)
                new_tags.append(tag)
            new_ner.append(new_tags)

        col1 = pd.DataFrame(pd.Series(docs), columns=[text_field])
        col2 = pd.DataFrame(pd.Series(new_ner), columns=[tags_field])
        corpus = pd.concat([col1, col2], axis=1)
        logger.warning("{} out of {} tags were ignored.".format(number_of_ignored_labels, number_of_labels))
        return corpus


def split_corpus(corpus, cutoff_train, cutoff_valid):
    """
    example: 
        cutoff_train 0.70
        cutoff_valid: 0.20
        --> test portion 10%
    """
    cutoff_train = round(len(corpus)*cutoff_train)
    cutoff_valid = round(len(corpus)*cutoff_valid)

    train_data = corpus[:cutoff_train]
    valid_data = corpus[cutoff_train:cutoff_train+cutoff_valid]
    test_data = corpus[cutoff_train+cutoff_valid:]

    logger.info("Train size {}, Valid size {}, Test size {}".format(len(train_data), len(valid_data), len(test_data)))

    return train_data, valid_data, test_data


if __name__ == "__main__":
    
    parser = ArgumentParser(description="Process corpus data")

    parser.add_argument(
        "--news_path", type=str, required=True, help="Path to tagged news data (Required) (es: /content/corpus/r*.json)"
    )
    parser.add_argument(
        "--reports_path", type=str, required=True, help="Path to tagged reports data (Required) (es: /content/corpus/r*.json)"
    )
    parser.add_argument(
        "--train_portion", type=float, default=0.7, help="default 0.7"
    )
    parser.add_argument(
        "--valid_portion", type=float, default=0.2, help="default 0.2"
    )
    parser.add_argument(
        "--binary_output_path", type=str, default = ".", help="default to local dir"
    )
    parser.add_argument(
        "--ents_to_remove", nargs="+", default=None, help="list of ents to remove. Example:\n APPLICATION HARDWARE (default: None)"
    )
    parser.add_argument(
        '--ents_to_rename', type=str, default=None, help="""dictionary of ents to modify. Example:\n '{"APPLICATION" : "PRODUCT", "HARDWARE" : "PRODUCT"}' (default: None)"""
    )
    parser.add_argument(
        '--aug_perc', type=float, default=None, help="Percentace of training data to augment, from 0 to 1. (default: None)"
    )
    parser.add_argument(
        '--count_ents', type = str, default = None, help=' <show> to show dataframe with entities count, <save> to show and save in current directory'
    )

    args = parser.parse_args()

    train_parameters = vars(args)

    print(train_parameters)


    news = read_json_files(args.news_path)
    reports = read_json_files(args.reports_path)

    # random.shuffle(news)
    # random.shuffle(reports)
    corpus = news+reports
    random.shuffle(corpus)

    # news_train, news_valid, news_test = split_corpus(news, args.train_portion, args.valid_portion)
    reports_train, reports_valid, reports_test = split_corpus(corpus, args.train_portion, args.valid_portion)
    
    if args.count_ents is not None:
        make_df_count(reports_train, reports_valid, reports_test, save = args.count_ents)

    # logger.debug(type(reports_train))

    #train_data = news_train + reports_train
    #valid_data = news_valid + reports_valid
    #test_data = news_test + reports_test

    # random.shuffle(train_data)
    # random.shuffle(valid_data)
    # random.shuffle(test_data)

    # Save data in .json format

    # Save data in .spacy format
    json_to_spacy_binary(reports_train, args.binary_output_path + "/training_data.spacy", SPECIAL_CASES, ents_to_remove = args.ents_to_remove, ents_to_rename = args.ents_to_rename, aug_perc=args.aug_perc)
    json_to_spacy_binary(reports_valid, args.binary_output_path + "/valid_data.spacy", SPECIAL_CASES, ents_to_remove = args.ents_to_remove, ents_to_rename = args.ents_to_rename)
    json_to_spacy_binary(reports_test, args.binary_output_path + "/test_data.spacy", SPECIAL_CASES, ents_to_remove = args.ents_to_remove, ents_to_rename = args.ents_to_rename)
