import numpy as np
import pandas as pd
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
    # print("hi")
    df = pd.read_csv(OPTS.input, header=None,
                           names=['idx', 'sentence', 'contains', 'probability'])
    #
    df.probability = df.probability.astype(float)
    # print(df.shape)
    print(f"Analyzing: {OPTS.input.split('/')[-1]}")
    # print(df.groupby('contains').mean(numeric_only=True))
    means = df.groupby('contains').mean(numeric_only=True)['probability']
    stderrs = df.groupby('contains').std(numeric_only=True)['probability'] / np.sqrt(len(df))
    print(f"contains: False    {means[False]:.4f} +- {stderrs[False]:.4f}")
    print(f"contains: True     {means[True]:.4f} +- {stderrs[True]:.4f}")
    print(f"Average prompt without comma = {df.loc[df.contains == False].sentence.map(lambda x: len(x)).mean()}")
    print(f"Average prompt with comma    = {df.loc[df.contains].sentence.map(lambda x: len(x)).mean()}")


if __name__ == '__main__':
    OPTS = parse_args()
    main()
