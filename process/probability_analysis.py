import numpy as np
import pandas as pd
import argparse

OPTS = None

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input1',
        required=True,
        help="the name of the first data file"
    )

    parser.add_argument(
        '--input2',
        required=True,
        help="the name of the second data file"
    )

    return parser.parse_args()

def main():
    df1 = pd.read_csv(OPTS.input1, header=None,
                           names=['idx', 'sentence', 'contains', 'probability'])
    print(df1.shape)
    df1['length'] = df1.sentence.str.len()

    df1.probability = df1.probability.astype(float)
    print()
    print(f"Analyzing: {OPTS.input1.split('/')[-1]}")
    means = df1.groupby('contains').mean(numeric_only=True)['probability']
    stderrs = df1.groupby('contains').std(numeric_only=True)['probability'] / np.sqrt(len(df1))
    print(f"contains: False    {means[False]:.4f} +- {stderrs[False]:.4f}")
    print(f"contains: True     {means[True]:.4f} +- {stderrs[True]:.4f}")
    without_df1 = df1[~df1.contains]
    print(f"Average prompt without comma = {df1[~df1.contains]['length'].mean()} +- {without_df1['length'].std() / np.sqrt(len(without_df1))}")
    with_df1 = df1[df1.contains]
    print(f"Average prompt with comma    = {with_df1['length'].mean()} +- {with_df1['length'].std() / np.sqrt(len(without_df1))}")
    print(f"Final = {df1.contains.value_counts()}")

    df2 = pd.read_csv(OPTS.input2, header=None,
                      names=['idx', 'sentence', 'contains', 'probability'])
    df2['length'] = df2.sentence.str.len()

    df2.probability = df2.probability.astype(float)
    print()
    print(f"Analyzing: {OPTS.input2.split('/')[-1]}")
    means = df2.groupby('contains').mean(numeric_only=True)['probability']
    stderrs = df2.groupby('contains').std(numeric_only=True)['probability'] / np.sqrt(len(df2))
    print(f"contains: False    {means[False]:.4f} +- {stderrs[False]:.4f}")
    print(f"contains: True     {means[True]:.4f} +- {stderrs[True]:.4f}")
    without_df2 = df2[~df2.contains]
    print(
        f"Average prompt without comma = {df2[~df2.contains]['length'].mean()} +- {without_df2['length'].std() / np.sqrt(len(without_df2))}")
    with_df2 = df2[df2.contains]
    print(
        f"Average prompt with comma    = {with_df2['length'].mean()} +- {with_df2['length'].std() / np.sqrt(len(without_df2))}")
    print(f"Final = {df2.contains.value_counts()}")

if __name__ == '__main__':
    OPTS = parse_args()
    main()
