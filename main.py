import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import mplcursors

matplotlib.use('TkAgg')
pd.set_option('display.max_colwidth', None)


class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None

    def load(self):
        try:
            df = pd.read_csv(self.filepath, sep='\t', encoding='cp1250')
            df['Win rate'] = pd.to_numeric(df['Win rate'], errors='coerce')
            self.df = df
            return df
        except Exception as e:
            messagebox.showerror("Loading error", f"Failed to load file:\n{e}")
            return None

    def validate_columns(self, required: list) -> bool:
        if self.df is None:
            return False
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            messagebox.showwarning(
                "Missing data",
                f"Required columns are missing in the file: {', '.join(missing)}"
            )
            return False
        return True


class Analyzer:
    CLUSTER_NAMES = {0: 'Support (low Elims)', 1: 'Tank', 2: 'DPS (high Elims)'}

    @staticmethod
    def _add_cursor(scatter, df, label_col: str = 'Hero') -> None:
        """Attaches mplcursors tooltip to a scatter plot."""
        cursor = mplcursors.cursor(scatter, hover=True)

        @cursor.connect("add")
        def on_add(sel):
            sel.annotation.set_text(df.iloc[sel.index][label_col])

    def cluster_analysis(self, df):
        required = ['Obj kills', 'OV s']
        if not all(col in df.columns for col in required):
            messagebox.showwarning("Missing data", "Required columns are missing in the file.")
            return None, []

        x = df[required]
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        kmeans = KMeans(n_clusters=3, random_state=42)
        df = df.copy()
        df['Cluster'] = kmeans.fit_predict(x_scaled)
        df['Group'] = df['Cluster'].map(self.CLUSTER_NAMES)
        df['Color'] = df['Cluster']

        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df['Obj kills'], df['OV s'], c=df['Color'], cmap='viridis', s=100)
        ax.set_title('Elims on objective vs. overall objective time [s]')
        ax.set_xlabel('Eliminations')
        ax.set_ylabel('Time [s]')

        handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=10)
            for i in range(3)
        ]
        ax.legend(handles, self.CLUSTER_NAMES.values(), title='Groups (Clusters)')
        self._add_cursor(scatter, df)

        info = [
            "The chart shows the division of characters into 3 clusters based on "
            "eliminations on objective and objective time.",
            "Clusters mostly overlap with hero roles.",
            "Hover over a point to see the hero's name.",
            ""
        ]
        return fig, info

    def support_efficiency(self, df):
        required = ['Role', 'KDA', 'H', 'Win rate', 'Hero']
        if not all(col in df.columns for col in required):
            messagebox.showwarning("Missing data", "Required columns are missing in the file.")
            return None, []

        df = df.copy()
        df['KDAH'] = 0.0
        df.loc[df['Role'] == 'Support', 'KDAH'] = (df['KDA'] * df['H'] / 1000).astype(float)
        df_support = df[df['Role'] == 'Support']

        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df_support['Win rate'], df_support['KDAH'], s=100, c='blue')
        sns.regplot(
            x='Win rate', y='KDAH', data=df_support,
            scatter_kws={'s': 100, 'color': 'blue'},
            line_kws={'color': 'red'}, ax=ax
        )
        ax.set_title('KDAH vs. Win rate')
        ax.set_xlabel('Win rate [%]')
        ax.set_ylabel('KDAH/1000')
        self._add_cursor(scatter, df_support)

        info = [
            "KDAH is a new parameter combining KDA and healing. "
            "The chart shows the relationship between efficiency and win rate for support characters.",
            "Hover over a point to see the hero's name.",
            "",
            ""
        ]
        return fig, info

    @staticmethod
    def fes_vs_efficiency(df):
        required = ['Elims', 'Solo kills', 'Final blows', 'Dmg', 'H', 'Hero']
        if not all(col in df.columns for col in required):
            messagebox.showwarning("Missing data", "Required columns are missing in the file.")
            return None, []

        df = df.copy()
        df['FES'] = df['Elims'] / (df['Solo kills'] + df['Final blows'])
        df['Eff'] = ((df['Dmg'] / df['Elims']) + (df['H'] / 1000)) / 100

        bins = np.linspace(df['Eff'].min(), df['Eff'].max(), 4)
        labels = ['Low Efficiency', 'Medium Efficiency', 'High Efficiency']
        df['Group'] = pd.cut(df['Eff'], bins=bins, labels=labels)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Group', y='FES', data=df, ax=ax)
        ax.set_title('FES vs efficiency group')
        ax.set_xlabel('Group')
        ax.set_ylabel('FES')

        grouped_heroes = (
            df.groupby('Group', observed=False)['Hero']
            .apply(lambda x: ', '.join(x))
            .reset_index()
        )
        info_rows = [f"{row['Group']}: {row['Hero']}" for _, row in grouped_heroes.iterrows()]

        info = [
            "FES is the ratio of eliminations to the sum of solo kills and final blows, "
            "shown in relation to efficiency.",
            info_rows[0] if len(info_rows) > 0 else "",
            info_rows[1] if len(info_rows) > 1 else "",
            info_rows[2] if len(info_rows) > 2 else ""
        ]
        return fig, info


class PlotManager:
    def __init__(self, preview_frame):
        self.preview_frame = preview_frame

    def show(self, fig) -> None:
        """Replaces any existing plot with the new figure."""
        for widget in self.preview_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.preview_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Game Data Analysis")
        self.analyzer = Analyzer()
        self._build_ui()

    def _build_ui(self) -> None:
        tk.Label(self, text="File name:").grid(row=0, column=0, padx=10, pady=0, sticky="w")
        self.file_entry = tk.Entry(self)
        self.file_entry.insert(0, "data.txt")
        self.file_entry.grid(row=1, column=0, padx=10, pady=0, sticky="w")

        tk.Button(self, text="Objective Clusters",
                  command=self._run_cluster_analysis).grid(row=4, column=0, padx=10, pady=10, sticky="w")
        tk.Button(self, text="Support Efficiency",
                  command=self._run_support_efficiency).grid(row=5, column=0, padx=10, pady=10, sticky="w")
        tk.Button(self, text="FES vs Efficiency",
                  command=self._run_fes_vs_efficiency).grid(row=6, column=0, padx=10, pady=10, sticky="w")
        tk.Button(self, text="Close",
                  command=self.quit).grid(row=13, column=0, padx=10, pady=10, sticky="w")

        preview_frame = tk.Frame(self, width=1140, height=600)
        preview_frame.grid(row=0, column=1, rowspan=14, padx=10, pady=10)
        preview_frame.grid_propagate(False)
        self.plot_manager = PlotManager(preview_frame)

        self.info_labels = []
        for i, row_num in enumerate(range(9, 13)):
            lbl = tk.Label(self, wraplength=260, justify="left", text=" ")
            lbl.grid(row=row_num, column=0, padx=10, pady=0, sticky="w")
            self.info_labels.append(lbl)

    def _load_data(self):
        """Loads data using DataLoader for the current file path."""
        loader = DataLoader(self.file_entry.get())
        return loader.load()

    def _set_info(self, lines: list) -> None:
        """Updates all four info labels from a list of strings."""
        for i, label in enumerate(self.info_labels):
            label.config(text=lines[i] if i < len(lines) else " ")

    def _display(self, fig, info: list) -> None:
        """Shows the figure and updates info labels; handles None fig gracefully."""
        if fig is None:
            return
        self._set_info(info)
        self.plot_manager.show(fig)

    def _run_cluster_analysis(self) -> None:
        df = self._load_data()
        if df is not None:
            fig, info = self.analyzer.cluster_analysis(df)
            self._display(fig, info)

    def _run_support_efficiency(self) -> None:
        df = self._load_data()
        if df is not None:
            fig, info = self.analyzer.support_efficiency(df)
            self._display(fig, info)

    def _run_fes_vs_efficiency(self) -> None:
        df = self._load_data()
        if df is not None:
            fig, info = self.analyzer.fes_vs_efficiency(df)
            self._display(fig, info)


if __name__ == "__main__":
    app = App()
    app.mainloop()
