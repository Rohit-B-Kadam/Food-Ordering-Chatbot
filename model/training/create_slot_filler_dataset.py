import json
import os
from path_manager import PATHS

MAIN_DATASET_PATHS = PATHS["dataset"]
UTTERANCES_FILE = "model/training/training_data/utterances.txt"
BIO_TAGS_FILE = "model/training/training_data/bio_tags.txt"
INTENT_FILE = "model/training/training_data/intent.txt"

UTTERANCES = ""
BIO_TAGS = ""
INTENT = ""


def write_in_file():
    with open(UTTERANCES_FILE, "w") as fd:
        fd.write(UTTERANCES)

    with open(BIO_TAGS_FILE, "w") as fd:
        fd.write(BIO_TAGS)

    with open(INTENT_FILE, "w") as fd:
        fd.write(INTENT)


def save_entries(text, bio_tags, intent ):
    global UTTERANCES, BIO_TAGS, INTENT
    UTTERANCES += text + "\n"
    BIO_TAGS += bio_tags + '\n'
    INTENT += intent + '\n'


def get_tag(index, segments):
    tag = "O"  # default value

    for segment in segments:
        annotation = segment["annotations"][0]["name"]
        if index == segment["start_index"]:
            tag = "B-" + annotation
        elif segment["start_index"] < index < segment["end_index"]:
            tag = "I-" + annotation
    return tag


def bio_tagger(text: str, segments: []) -> str:
    tokens = text.split(" ")
    tags = []
    index = 0
    for token in tokens:
        tag = get_tag(index, segments)
        tags.append(tag)
        index += len(token) + 1  # plus 1 for space
    return " ".join(tags)   # return string


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
                tags = bio_tagger(text, segments)
                save_entries(text, tags, intent)
            else:
                print("No segment")

    # write to the file
    write_in_file()


if __name__ == "__main__":
    create_bio_dataset()
