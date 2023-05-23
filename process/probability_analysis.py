import pandas as pd
# import seaborn as sns
import argparse

OPTS = None

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input',
        required=True,
        help="the name of the directory that stores the data to be scored"
    )

    return parser.parse_args()

def main():
    df = pd.read_csv(OPTS.input, header=None,
                           names=['idx', 'sentence', 'contains', 'probability'])

    df.probability = df.probability.astype(float)

    print(f"Analyzing: {OPTS.input.split('/')[-1]}")
    print(df.groupby('contains').mean(numeric_only=True))
    print(
        f"Average prompt without comma = {df.loc[df.contains == False].sentence.map(lambda x: len(x)).mean()}")
    print(
        f"Average prompt with comma  = {df.loc[df.contains].sentence.map(lambda x: len(x)).mean()}")


if __name__ == '__main__':
    OPTS = parse_args()
    main()
