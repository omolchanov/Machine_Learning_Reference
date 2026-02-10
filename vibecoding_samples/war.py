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


class WarLossesAnalyzer:
    def __init__(self, api_url: str = "https://russian-casualties.in.ua/api/v1/data/json/daily"):
        self.api_url = api_url
        self.payload = self.fetch_daily_losses()
        self.df = self.to_dataframe(self.payload)

    def fetch_daily_losses(self) -> Dict[str, Any]:
        try:
            resp = requests.get(self.api_url, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch data from API: {e}") from e

    def to_dataframe(self, payload: Dict[str, Any]) -> pd.DataFrame:
        data = payload.get("data", {})
        df = (
            pd.DataFrame.from_dict(data, orient="index")
            .rename_axis("date")
            .reset_index()
        )
        df["date"] = pd.to_datetime(df["date"], format="%Y.%m.%d")
        df = df.set_index("date").sort_index()
        return df

    def inspect_df(self, n: int = 5) -> None:
        df = self.df
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

    def plot_6month_losses(self, columns: list[str] | None = None, figsize: tuple[int, int] = (12, 6)) -> None:
        df = self.df
        if columns is None:
            columns = ["personnel", "missiles", "uav"]
        data = df.select_dtypes("number")[columns]
        agg = data.resample("6MS").sum()
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
            for x, y in zip(agg.index, agg[col]):
                ax.annotate(
                    f"{int(y):,}",
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

    def correlation_analysis(self, columns: list[str] = ["personnel", "uav", "missiles"], plot: bool = False) \
            -> pd.DataFrame:
        data = self.df[columns].dropna()
        corr_matrix = data.corr()
        if plot:
            plt.figure(figsize=(6, 4))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Correlation Matrix")
            plt.tight_layout()
            plt.show()
        return corr_matrix

    def run_analysis(self):
        self.inspect_df()
        self.plot_6month_losses(['personnel', 'uav', 'missiles'])

        print("\nCorrelation matrix between personnel, uav, and missiles:")
        print(self.correlation_analysis())


if __name__ == "__main__":
    analyzer = WarLossesAnalyzer()
    analyzer.run_analysis()
