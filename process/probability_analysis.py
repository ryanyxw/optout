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
    df = pd.read_csv(OPTS.input, header=None,
                           names=['idx', 'sentence', 'contains', 'probability'])
    df['length'] = df.sentence.str.len()

    df.probability = df.probability.astype(float)
    print()
    print(f"Analyzing: {OPTS.input.split('/')[-1]}")
    means = df.groupby('contains').mean()['probability'] 
    stderrs = df.groupby('contains').std()['probability'] / np.sqrt(len(df))
    print(f"contains: False    {means[False]:.4f} +- {stderrs[False]:.4f}")
    print(f"contains: True     {means[True]:.4f} +- {stderrs[True]:.4f}")
    without_df = df[~df.contains]
    print(without_df['length'].median())
    print(f"Average prompt without comma = {df[~df.contains]['length'].mean()} +- {without_df['length'].std() / np.sqrt(len(without_df))}")
    with_df = df[df.contains]
    print(with_df['length'].median())
    print(f"Average prompt with comma    = {with_df['length'].mean()} +- {with_df['length'].std() / np.sqrt(len(without_df))}")


if __name__ == '__main__':
    OPTS = parse_args()
    main()
