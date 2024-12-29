import pandas as pd


def indexData(input_path, output_path):
    df = pd.read_csv(input_path)
    df.insert(0, 'tno', range(len(df)))
    df.to_csv(output_path, index=False)


def headData(input_path, output_path, n):
    df = pd.read_csv(input_path)
    df_head = df.head(n)
    df_head.to_csv(output_path, index=False)


def separateCellChange(input_path, output_path):
    df = pd.read_csv(input_path, header=None)
    df[['key', 'value']] = df[0].str.split('.', expand=True)
    df.to_csv(output_path, index=False, header=False)


def countCellDifference(clean_path, dirty_path):
    df_dirty = pd.read_csv(dirty_path, dtype=str)
    df_clean = pd.read_csv(clean_path, dtype=str)
    differences = df_dirty.ne(df_clean)
    print(differences.sum().sum())


def countTupleDifference(clean_path, dirty_path):
    df_dirty = pd.read_csv(dirty_path, dtype=str)
    df_clean = pd.read_csv(clean_path, dtype=str)
    differences = df_dirty.ne(df_clean).any(axis=1)
    print(differences.sum())


if __name__ == '__main__':
    # indexData("tax_50k/clean_index_10k.csv", "tax_50k/clean_index_10k.csv")
    headData("../6_soccer/clean_index.csv", "../6_soccer/subset_directly_clean_index_10k.csv", 10000)
    # separateCellChange("tax_200k/dirty_mix_0.5/cellChanges.csv", "tax_200k/dirty_mix_0.5/cell_changes.csv")
    # countCellDifference("tax_200k/tax_200k_clean_id.csv",
    #                     "tax_200k/noise/dirty_mix_0.25/dirty_tax_mix_0.25.csv")
    # countTupleDifference("tax_200k/tax_200k_clean_id.csv",
    #                      "tax_200k/noise/dirty_mix_0.25/dirty_tax_mix_0.25.csv")
