import pandas as pd
import numpy as np


class CVAnalyzer:
    def __init__(self):
        self.df = pd.read_csv('../CV_Analytical_Portal/assets/cv-dataset.csv')

    def calculate_current_cases(self):
        self.df['Current Cases'] = self.df['Total Cases'] - self.df['Recoveries'] - self.df['Deaths']

    def calculate_totals(self):
        self.df.loc['Total', 'Total Cases'] = self.df['Total Cases'].sum()
        self.df.loc['Total', 'Recoveries'] = self.df['Recoveries'].sum()
        self.df.loc['Total', 'Deaths'] = self.df['Deaths'].sum()

        # TODO: Implement checking of empty variable
        self.df.loc['Total', 'Current Cases'] = self.df['Current Cases'].sum()

    def get_area_with_max_cases(self):
        return self.df.loc[self.df['Total Cases'].idxmax()][0]


class CVDashboard:

    def __init__(self):
        self.cva = CVAnalyzer()

    def _print_area_with_max_cases(self):
        print('Area with the maximum Total Cases is {}'.format(self.cva.get_area_with_max_cases()))

    def _print_df_with_totals(self):
        self.cva.calculate_current_cases()
        self.cva.calculate_totals()
        print(self.cva.df)

    def render_basic_dashboard(self):
        print('!------BASIC DASHBOARD-------!')
        self._print_area_with_max_cases()

    def render_advanced_dashboard(self):
        print('!------ADVANCED DASHBOARD-------!')
        self._print_df_with_totals()


cv_dashboard = CVDashboard()
cv_dashboard.render_basic_dashboard()
cv_dashboard.render_advanced_dashboard()



