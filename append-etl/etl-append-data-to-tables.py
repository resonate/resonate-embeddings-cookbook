import pandas as pd
import numpy as np
import os
import argparse
import sys
import logging

def split_svkeys(attributes):
    """Split the 'ATTRIBUTES' column into a list of svkeys."""
    if pd.isna(attributes):
        return []
    return [svkey.strip() for svkey in attributes.split(",")]

def map_svkey_to_attr(true_svkeys, attr, single_select_map):
    """
    Map svkeys to a single select attribute.
    Returns the first matching svkey or None.
    """
    for svkey in true_svkeys:
        if svkey in single_select_map and single_select_map[svkey] == attr:
            return svkey  # or another desired value from the mapping
    return None

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="ETL script to process dataAppend and taxonomy files and save as Parquet.")

    parser.add_argument(
        "--data_append_path",
        type=str,
        required=True,
        help="Path to your dataAppend file (csv.gz). Supports S3 URIs."
    )

    parser.add_argument(
        "--data_append_taxonomy_path",
        type=str,
        required=True,
        help="Path to your data append taxonomy file (csv). Supports S3 URIs."
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory where you want to save your output Parquet files. Supports S3 URIs."
    )

    return parser.parse_args()

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    args = parse_arguments()

    data_append_path = args.data_append_path
    data_append_taxonomy_path = args.data_append_taxonomy_path
    out_dir = args.out_dir

    # Ensure output directory exists (for S3, this step is not strictly necessary)
    # os.makedirs(out_dir, exist_ok=True)  # Commented out since S3 handles directories implicitly

    # Read in the data assets
    logging.info("Reading dataAppend...")
    try:
        data_append = pd.read_csv(data_append_path, compression='gzip', header=0, dtype=str)
    except Exception as e:
        logging.error(f"Failed to read dataAppend: {e}")
        sys.exit(1)

    logging.info("Reading dataAppendTaxonomy...")
    try:
        data_append_taxonomy = pd.read_csv(data_append_taxonomy_path, header=0, dtype=str)
    except Exception as e:
        logging.error(f"Failed to read dataAppendTaxonomy: {e}")
        sys.exit(1)

    # Process 'dataAppend' by splitting 'ATTRIBUTES' into 'trueSvkeys'
    logging.info("Processing 'dataAppend' by splitting 'ATTRIBUTES'...")
    data_append['trueSvkeys'] = data_append['ATTRIBUTES'].apply(split_svkeys)

    # Extract distinct svkeys
    logging.info("Extracting distinct svkeys...")
    super_set_svkeys = pd.Series([svkey for sublist in data_append['trueSvkeys'] for svkey in sublist]).unique().tolist()

    # Filter taxonomy based on the relevant svkeys and create 'attributeString'
    logging.info("Filtering taxonomy and creating 'attributeString'...")
    focal_taxonomy = data_append_taxonomy[data_append_taxonomy['survey_value_key'].isin(super_set_svkeys)].copy()
    focal_taxonomy['attributeString'] = focal_taxonomy['AttributeKey'] + "_" + focal_taxonomy['survey_value_key'].astype(str)

    # Create a map for single select attributes
    logging.info("Creating singleSelectColumnMap...")
    single_select = focal_taxonomy[focal_taxonomy['Attribute Type'] == "Single Select"]
    single_select_column_map = pd.Series(single_select.AttributeKey.values, index=single_select.survey_value_key.astype(str)).to_dict()

    # Extract distinct single and multi-select attributes
    logging.info("Extracting distinct single and multi-select attributes...")
    single_select_attributes = list(single_select_column_map.values())

    multi_select = focal_taxonomy[focal_taxonomy['Attribute Type'] == "Multi Select"]
    multi_select_attributes = multi_select['attributeString'].unique().tolist()

    all_attributes = focal_taxonomy['attributeString'].unique().tolist()

    # Create singleSelectDF by adding a new column for each single select attribute
    logging.info("Creating singleSelectDF...")
    for attr in single_select_attributes:
        data_append[attr] = data_append['trueSvkeys'].apply(lambda svkeys: map_svkey_to_attr(svkeys, attr, single_select_column_map))

    single_select_df = data_append.drop(columns=['trueSvkeys', 'ATTRIBUTES'])

    # Save singleSelectDF as Parquet
    logging.info("Saving singleSelectDF...")
    try:
        single_select_df.to_parquet(os.path.join(out_dir, "singleSelectDF.parquet"), index=False)
    except Exception as e:
        logging.error(f"Failed to save singleSelectDF: {e}")
        sys.exit(1)

    # Process multi-select attributes
    logging.info("Processing multi-select attributes...")
    # Explode the 'trueSvkeys'
    exploded = data_append.explode('trueSvkeys')
    exploded = exploded.rename(columns={'trueSvkeys': 'survey_value_key'})

    # Merge with focal_taxonomy on 'survey_value_key'
    merged = exploded.merge(focal_taxonomy, on='survey_value_key', how='left')

    # Create pivot table for multi-select attributes
    logging.info("Creating multiSelectDF...")
    multi_select_df = merged.pivot_table(
        index='ID',
        columns='attributeString',
        aggfunc='size',
        fill_value=0
    ).reset_index()

    # Ensure all multi_select_attributes are present
    for attr in multi_select_attributes:
        if attr not in multi_select_df.columns:
            multi_select_df[attr] = 0

    multi_select_df = multi_select_df[['ID'] + multi_select_attributes]

    # Save multiSelectDF as Parquet
    logging.info("Saving multiSelectDF...")
    try:
        multi_select_df.to_parquet(os.path.join(out_dir, "multiSelectDF.parquet"), index=False)
    except Exception as e:
        logging.error(f"Failed to save multiSelectDF: {e}")
        sys.exit(1)

    # Alternatively, process all attributes as multi-selects
    logging.info("Processing all attributes as multi-selects...")
    all_as_multi_select_df = merged.pivot_table(
        index='ID',
        columns='attributeString',
        aggfunc='size',
        fill_value=0
    ).reset_index()

    # Ensure all_attributes are present
    for attr in all_attributes:
        if attr not in all_as_multi_select_df.columns:
            all_as_multi_select_df[attr] = 0

    all_as_multi_select_df = all_as_multi_select_df[['ID'] + all_attributes]

    # Save allAsMultiSelectDF as Parquet
    logging.info("Saving allAsMultiSelectDF...")
    try:
        all_as_multi_select_df.to_parquet(os.path.join(out_dir, "allAsMultiSelectDF.parquet"), index=False)
    except Exception as e:
        logging.error(f"Failed to save allAsMultiSelectDF: {e}")
        sys.exit(1)

    logging.info("ETL process completed successfully.")

if __name__ == "__main__":
    main()
