# exoplanet_api/convert_csv_to_parquet.py

import pandas as pd

# Замените путь на свой CSV
csv_path = "data/combined_exoplanet_dataset_imputed.csv"
parquet_path = "data/combined_exoplanet_dataset_imputed.parquet"

# Чтение CSV по кускам и запись в Parquet
chunks = pd.read_csv(csv_path, chunksize=10000)

# Конкатим и сохраняем
df_list = []
for chunk in chunks:
    df_list.append(chunk)

df = pd.concat(df_list, ignore_index=True)
df.to_parquet(parquet_path, index=False)
print(f"Saved Parquet file: {parquet_path}")
