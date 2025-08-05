import pandas as pd
import numpy as np

def poison_data(poison_level, seed=42):
    """Poisons a percentage of the training data by flipping labels."""
    train_df = pd.read_csv('data/train.csv')
    num_to_poison = int(len(train_df) * poison_level)
    np.random.seed(seed)
    poison_indices = np.random.choice(train_df.index, num_to_poison, replace=False)

    for idx in poison_indices:
        original_label = train_df.loc[idx, 'species']
        possible_new_labels = [l for l in [0, 1, 2] if l != original_label]
        train_df.loc[idx, 'species'] = np.random.choice(possible_new_labels)

    train_df.to_csv('data/train.csv', index=False) # Overwrite the training file

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("poison_level", type=float, help="The percentage of data to poison (0.0 to 1.0)")
    args = parser.parse_args()
    poison_data(args.poison_level)