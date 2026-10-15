import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    match_columns = ["age", "gender", "zip3"]

    # Join records that have the same quasi-identifiers.
    possible_matches = anon_df.merge(aux_df, on=match_columns, how="inner")

    # A record is unique only when it has exactly one possible name.
    match_counts = possible_matches.groupby("anon_id")["name"].transform("size")
    unique_matches = possible_matches[match_counts == 1]

    return unique_matches[["anon_id", "name"]].rename(
        columns={"name": "matched_name"}
    )


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    if len(anon_df) == 0:
        return 0.0

    return len(matches_df) / len(anon_df)
