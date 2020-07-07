import json
import pandas as pd
import spacy
from path_manager import PATHS

MAIN_DATASET_PATHS = PATHS["dataset"]
UTTERANCES_FILE = PATHS["slot_tagger_dataset"]["utterances"]
BIO_TAGS_FILE = PATHS["slot_tagger_dataset"]["bio_tags"]
INTENT_FILE = PATHS["slot_tagger_dataset"]["intent"]

UTTERANCES = []
BIO_TAGS = []
INTENT = []

NLP = spacy.load('en_core_web_sm')
DFLIST = []


def write_in_file():
    with open(UTTERANCES_FILE, "w") as fd:
        fd.write("\n".join(UTTERANCES))

    with open(BIO_TAGS_FILE, "w") as fd:
        fd.write("\n".join(BIO_TAGS))

    with open(INTENT_FILE, "w") as fd:
        fd.write("\n".join(INTENT))


def save_entries(text: str, bio_tags: str, intent: str):
    global UTTERANCES, BIO_TAGS, INTENT
    UTTERANCES.append(text)
    BIO_TAGS.append(bio_tags)
    INTENT.append(intent)


def get_tag(index, segments):
    tag = "O"  # default value
    if not segments:
        return tag

    for segment in segments:
        annotation = segment["annotations"][0]["name"]
        if index == segment["start_index"]:
            tag = "B-" + annotation
        elif segment["start_index"] < index < segment["end_index"]:
            tag = "I-" + annotation
    return tag


def bio_tagger(text: str, segments: [] = None):
    doc = NLP(text)
    index = 0
    tags = []
    tokens = []
    for token in doc:
        tag = get_tag(index, segments)
        tags.append(tag)
        tokens.append(token.text)
        index += len(token.text)
        if index != len(text) and text[index] == " ":  # if there is space
            index += 1

    # DFLIST.append(tokens)
    # DFLIST.append(tags)
    tokens = " ".join(tokens)
    tags = " ".join(tags)
    return tokens, tags


def extract_intent(instruction_id: str):
    intent = instruction_id.rsplit("-", 1)[0]
    return intent


def create_bio_dataset():
    dataset = json.load(open(MAIN_DATASET_PATHS, "r"))
    print(len(dataset))

    for data in dataset:
        utterances = data["utterances"]
        intent = extract_intent(data["instruction_id"])
        for utterance in utterances:
            text = utterance["text"]
            segments = utterance.get("segments")
            if segments:
                tokens, tags = bio_tagger(text, segments)
            else:
                tokens, tags = bio_tagger(text)
            save_entries(tokens, tags, intent)

    # df = pd.DataFrame(DFLIST)
    # df.to_csv("model/training/training_data/check.csv")
    # write to the file
    write_in_file()


# def check_spacy():
#
#     doc = nlp("This's great")
#     print([token.text for token in doc])
#     print([token.lemma_ for token in doc])


def main():
    create_bio_dataset()


if __name__ == "__main__":
    main()
