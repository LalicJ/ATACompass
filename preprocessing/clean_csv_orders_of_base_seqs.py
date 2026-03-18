import pandas as pd
import os

os.makedirs("cleaned_csv", exist_ok=True)


def normalize_case(value):
    if isinstance(value, str):
        return value.lower().replace("cells", "").replace("cell", "").strip()
    return value

csv_list = [
    (
        [
            "20250212_cross_batch_data_processed/training_set/Human1_heart.csv",
        ],
        [
            ("20250212_cross_batch_data_processed/prediction_set/Human2_heart_combined.csv", "Human2_heart_combined.csv"),
        ],
    ),
]

for idx, (csv_file1_list, csv_file2_list) in enumerate(csv_list):
    print(csv_file1_list, csv_file2_list)

    df_list_1 = []
    for path in csv_file1_list:
        df = pd.read_csv(path, encoding="utf-8")  # 根据需要调整编码
        df_list_1.append(df)

    df1 = pd.concat(df_list_1, ignore_index=False)

    csv_file2_list = [i[0] for i in csv_file2_list]
    df_list_2 = []
    for path in csv_file2_list:
        df = pd.read_csv(path, encoding="utf-8") 
        df_list_2.append(df)

    df2 = pd.concat(df_list_2, ignore_index=False)

    column1 = "cell_type"  # 替换为实际的列名

    df1["Normalized"] = df1[column1].apply(normalize_case)

    count1 = df1["Normalized"].value_counts().reset_index()
    count1.columns = ["Normalized", "Count"]

    column2 = "cell_type"  # 替换为实际的列名

    df2["Normalized"] = df2[column2].apply(normalize_case)

    count2 = df2["Normalized"].value_counts().reset_index()
    count2.columns = ["Normalized", "Count"]

    common_values = set(count1["Normalized"]).intersection(set(count2["Normalized"]))

    filtered_df1 = df1[df1["Normalized"].isin(common_values)].drop(
        columns=["Normalized"]
    )
    filtered_df1.to_csv(f'cleaned_csv/raw_{idx}_{csv_file1_list[0].replace("/", "_")}')

    for csv_file2, df2_tmp in zip(csv_file2_list, df_list_2):
        df2_tmp["Normalized"] = df2_tmp[column2].apply(normalize_case)
        filtered_df2 = df2_tmp[df2_tmp["Normalized"].isin(common_values)].drop(
            columns=["Normalized"]
        )

        filtered_df2.to_csv(f'cleaned_csv/{csv_file2.replace("/", "_")}')
