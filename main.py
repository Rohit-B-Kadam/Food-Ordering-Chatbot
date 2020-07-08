import sys
from model.training.train_slot_tagger import train_model, evaluate_model, print_help


def main():
    if len(sys.argv) == 2:
        if sys.argv[1].lower() == "--train":
            train_model()
        elif sys.argv[1].lower() == "--eval":
            evaluate_model()
        else:
            print_help()
    else:
        print_help()


if __name__ == "__main__":
    main()
