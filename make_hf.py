"""
script for creating test,train datasets with all 3 models
pip install polars==0.19 sciris
"""

from pathlib import Path
import polars as pl
import sciris as sc
import numpy as np
rng = np.random.default_rng(seed=12345) # for reproducibility

# current directory
this_dir = Path(__file__).parent

# load the datasets (original set)
df = None
for split in ['test', 'train']:
    for model in ['gpt4', 'llama2']:
        for cat in range(1,10):
            df_ = pl.read_csv(this_dir / "datasets" / split / model / f"med_safety_demonstrations_category_{cat}.csv")
            # add columns about model
            df_ = df_.with_columns(pl.lit(model).alias('model'))
            # add column about split
            df_ = df_.with_columns(pl.lit(split).alias('split'))
            # add column about category
            df_ = df_.with_columns(pl.lit(cat).alias('cat'))
            if df is None:
                df = df_
            else:
                df = pl.concat((df, df_))
df = df.drop(['', 'safe_response'])
print("original dataset shape:", df.shape)
print(df.group_by('split').count())

# add llama3 results
for cat in range(1,10):
    with open(this_dir / "datasets" / "med_harm_llama3" / f"category_{cat}.txt", "r") as f:
        lines = f.readlines()
    df_ = pl.DataFrame({'harmful_medical_request': lines})
    # add columns about model
    df_ = df_.with_columns(pl.lit('llama3').alias('model'))
    # add column about split
    df_ = df_.with_columns(pl.lit('train').alias('split'))
    # add column about category
    df_ = df_.with_columns(pl.lit(cat).alias('cat'))
# clean for tokens
df_ = df_.with_columns(pl.col('harmful_medical_request').str.replace('<|eot_id|>', ''))
# clean with strip
df_ = df_.with_columns(pl.col('harmful_medical_request').str.strip_chars())
# maintain 50:50 test:train split
df_ = df_.with_columns(pl.struct('split').map_elements(lambda s: 'train' if rng.random() < 0.5 else 'test', return_dtype=pl.String))
df = pl.concat((df, df_))

# writeout to test and train csv
for split in ['test', 'train']:
    df_split = df.filter(pl.col('split') == split)
    file_name = this_dir / 'hf' / f'{split}.csv'
    sc.makefilepath(file_name, makedirs=True)
    df_split.write_csv(file_name)
    print(f'wrote {df_split.shape} to hf/{split}.csv')

print("done")

