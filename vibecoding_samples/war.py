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

    if columns is None:
        columns = ["personnel", "missiles", "uav"]

        # numeric only
    data = df.select_dtypes("number")[columns]

    # 6-month aggregation
    agg = data.resample("6MS").sum()

    # create subplots (one per metric)
    fig, axes = plt.subplots(len(columns), 1, figsize=figsize, sharex=True)

    if len(columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        sns.lineplot(
            x=agg.index,
            y=agg[col],
            marker="o",
            ax=ax
        )

        ax.set_title(col.capitalize())
        ax.set_ylabel("losses")

        # annotate each point
        for x, y in zip(agg.index, agg[col]):
            ax.annotate(
                f"{int(y):,}",  # formatted with commas
                (x, y),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=9
            )

        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Period (6 months)")

    plt.tight_layout()
    plt.show()


def correlation_analysis(df: pd.DataFrame, columns: list[str] = ["personnel", "uav", "missiles"], plot: bool = False) -> pd.DataFrame:
    """
    Calculate and optionally plot the correlation matrix between selected features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the features.
    columns : list[str]
        List of feature names to include in the correlation analysis.
    plot : bool
        Whether to plot a heatmap of the correlation matrix.

    Returns
    -------
    pd.DataFrame
        Correlation matrix between the selected features.
    """
    # Select only the specified columns and drop rows with missing values
    data = df[columns].dropna()
    corr_matrix = data.corr()

    if plot:
        plt.figure(figsize=(6, 4))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.show()

    return corr_matrix


def main() -> None:
    payload = fetch_daily_losses()
    df = to_dataframe(payload)

    # inspect_df(df)
    plot_6month_losses(df, ['personnel', 'uav', 'missiles'])

    # Correlation analysis demonstration
    corr = correlation_analysis(df)
    print("\nCorrelation matrix between personnel, uav, and missiles:")
    print(corr)


if __name__ == "__main__":
    main()