from typing import Dict, Any

import pandas as pd
import requests

import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1)


API_URL = "https://russian-casualties.in.ua/api/v1/data/json/daily"


def fetch_daily_losses() -> Dict[str, Any]:
    try:
        resp = requests.get(API_URL, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch data from API: {e}") from e


def to_dataframe(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert API payload to a pandas DataFrame.

    Index: datetime (parsed from 'YYYY.MM.DD')
    Columns: equipment/personnel fields from the 'data' dict.
    """
    data = payload.get("data", {})

    # Build DataFrame: rows = dates, columns = metrics
    df = (
        pd.DataFrame.from_dict(data, orient="index")
        .rename_axis("date")
        .reset_index()
    )

    # Convert 'date' string like '2022.02.24' to proper datetime
    df["date"] = pd.to_datetime(df["date"], format="%Y.%m.%d")
    df = df.set_index("date").sort_index()
    return df


def inspect_df(df: pd.DataFrame, n: int = 5) -> None:
    """
    Quick exploratory summary of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect
    n : int
        Number of rows for head/tail
    """
    print("\n" + "=" * 60)
    print("SHAPE")
    print("=" * 60)
    print(df.shape)

    print("\n" + "=" * 60)
    print("COLUMNS")
    print("=" * 60)
    print(list(df.columns))

    print("\n" + "=" * 60)
    print("DTYPES")
    print("=" * 60)
    print(df.dtypes)

    print("\n" + "=" * 60)
    print("INFO")
    print("=" * 60)
    df.info()

    print("\n" + "=" * 60)
    print(f"HEAD ({n})")
    print("=" * 60)
    print(df.head(n))

    print("\n" + "=" * 60)
    print(f"TAIL ({n})")
    print("=" * 60)
    print(df.tail(n))

    print("\n" + "=" * 60)
    print("DESCRIBE (numeric)")
    print("=" * 60)
    print(df.describe())

    print("\n" + "=" * 60)
    print("DESCRIBE (all)")
    print("=" * 60)
    print(df.describe(include="all"))


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_6month_losses(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> None:
    """
    Aggregate daily losses by 6-month periods and show a seaborn bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame indexed by date with numeric metrics
    columns : list[str] | None
        Which metrics to plot (default = all numeric)
    figsize : tuple[int, int]
        Figure size
    """

    # Keep numeric columns
    numeric = df.select_dtypes("number")
    if columns:
        numeric = numeric[columns]

        # Aggregate by 6-month periods (starting from Jan/Jul)
    agg = numeric.resample("6MS").sum()

    # Convert to long format for seaborn
    plot_df = agg.reset_index().melt(
        id_vars="date", var_name="metric", value_name="losses"
    )

    # Plot
    plt.figure(figsize=figsize)
    ax = sns.barplot(data=plot_df, x="date", y="losses", hue="metric")
    plt.xticks(rotation=45)
    plt.title("Losses aggregated by 6-month periods")

    # Annotate bars with values
    for p in ax.patches:
        height = p.get_height()
        if height > 0:  # avoid zero-height bars
            ax.annotate(
                f"{int(height)}",
                (p.get_x() + p.get_width() / 2., height),
                ha='center',
                va='bottom',
                fontsize=9,
                xytext=(0, 3),
                textcoords='offset points'
            )

    plt.tight_layout()
    plt.show()

def main() -> None:
    payload = fetch_daily_losses()
    df = to_dataframe(payload)

    # inspect_df(df)
    plot_6month_losses(df, ['personnel', 'uav', 'missiles'])


if __name__ == "__main__":
    main()

