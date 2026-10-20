"""Validate the raw Telco data before cleaning or model training."""

from typing import List, Tuple

import great_expectations as gx
import pandas as pd


def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    required = [
        "customerID", "gender", "Partner", "Dependents", "PhoneService",
        "InternetService", "Contract", "tenure", "MonthlyCharges",
        "TotalCharges", "Churn",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        return False, [f"Missing column: {column}" for column in missing]

    # The source CSV stores eleven new customers' TotalCharges as blanks.
    # Treat those as missing; reject any other value that cannot be parsed.
    checked = df.copy()
    raw_charges = checked["TotalCharges"].fillna("").astype(str).str.strip()
    checked["TotalCharges"] = pd.to_numeric(raw_charges, errors="coerce")
    invalid_charges = raw_charges.ne("") & checked["TotalCharges"].isna()
    if invalid_charges.any():
        return False, [f"Invalid TotalCharges values: {int(invalid_charges.sum())}"]

    context = gx.get_context(mode="ephemeral")
    source = context.data_sources.add_pandas(name="telco")
    asset = source.add_dataframe_asset(name="customers")
    batch_definition = asset.add_batch_definition_whole_dataframe("all_customers")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": checked})

    suite = gx.ExpectationSuite(name="telco_data_quality")
    for column in required:
        suite.add_expectation(gx.expectations.ExpectColumnToExist(column=column))
    for column in ("customerID", "tenure", "MonthlyCharges", "Churn"):
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=column))

    allowed_values = {
        "gender": ["Male", "Female"],
        "Partner": ["Yes", "No"],
        "Dependents": ["Yes", "No"],
        "PhoneService": ["Yes", "No"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "Churn": ["Yes", "No"],
    }
    for column, values in allowed_values.items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(column=column, value_set=values)
        )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(column="tenure", min_value=0, max_value=120)
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(column="MonthlyCharges", min_value=0, max_value=200)
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(column="TotalCharges", min_value=0)
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnPairValuesAToBeGreaterThanB(
            column_A="TotalCharges", column_B="MonthlyCharges", or_equal=True,
            mostly=0.95, ignore_row_if="either_value_is_missing",
        )
    )

    result = batch.validate(suite)
    failures = [item.expectation_config.type for item in result.results if not item.success]
    print(f"Data validation: {len(result.results) - len(failures)}/{len(result.results)} checks passed")
    return bool(result.success), failures
