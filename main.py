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

# --- functions ---

def load_file():
    """loads the file specified by user"""
    filename = file_entry.get()
    try:
        df = pd.read_csv(filename, sep='\t', encoding='cp1250')
        df['Win rate'] = pd.to_numeric(df['Win rate'], errors='coerce')
        return df
    except Exception as e:
        messagebox.showerror("Loading error", f"Failed to load file:\n{e}")
        return None

def show_plot(fig):
    """displays the plot in the application window"""
    for widget in preview_frame.winfo_children():
        widget.destroy()
    canvas = FigureCanvasTkAgg(fig, master=preview_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

def close_app():
    root.quit()

def add_cursor(scatter, df, label_col='Hero'):
    """adds tooltips when hovering over plot points"""
    cursor = mplcursors.cursor(scatter, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        index = sel.index
        sel.annotation.set_text(df.iloc[index][label_col])

# --- Analyses ---

def cluster_analysis():
    """cluster analysis based on eliminations and objective time"""
    df = load_file()
    if df is None:
        return
    required_cols = ['Obj kills', 'OV s']
    if not all(col in df.columns for col in required_cols):
        messagebox.showwarning("Missing data", "Required columns are missing in the file.")
        return

    X = df[required_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # assign cluster names
    cluster_names = {0: 'Support (low Elims)', 1: 'Tank', 2: 'DPS (high Elims)'}
    df['Group'] = df['Cluster'].map(cluster_names)
    df['Color'] = df['Cluster']

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df['Obj kills'], df['OV s'], c=df['Color'], cmap='viridis', s=100)
    ax.set_title('Elims on objective vs. overall objective time [s]')
    ax.set_xlabel('Eliminations')
    ax.set_ylabel('Time [s]')
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=10)
               for i in range(3)]
    ax.legend(handles, cluster_names.values(), title='Groups (Clusters)')
    add_cursor(scatter, df)
    set_info(
        "The chart shows the division of characters into 3 clusters based on eliminations on objective and objective time.",
        "Clusters mostly overlap with hero roles.",
        "Hover over a point to see the hero's name.",
        ""
    )
    show_plot(fig)

def support_efficiency():
    """analysis of support efficiency (KDAH vs win rate)"""
    df = load_file()
    if df is None:
        return
    if not all(col in df.columns for col in ['Role', 'KDA', 'H', 'Win rate', 'Hero']):
        messagebox.showwarning("Missing data", "Required columns are missing in the file.")
        return

    df['KDAH'] = 0.0
    df.loc[df['Role'] == 'Support', 'KDAH'] = (df['KDA'] * df['H'] / 1000).astype(float)
    df_support = df[df['Role'] == 'Support']

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df_support['Win rate'], df_support['KDAH'], s=100, c='blue')
    sns.regplot(x='Win rate', y='KDAH', data=df_support, scatter_kws={'s': 100, 'color': 'blue'},
                line_kws={'color': 'red'}, ax=ax)
    ax.set_title('KDAH vs. Win rate')
    ax.set_xlabel('Win rate [%]')
    ax.set_ylabel('KDAH/1000')
    add_cursor(scatter, df_support)
    set_info(
        "KDAH is a new parameter combining KDA and healing. The chart shows the relationship between efficiency and win rate for support characters.",
        "Hover over a point to see the hero's name.",
        "",
        ""
    )
    show_plot(fig)

def fes_vs_efficiency():
    """analysis of FES in relation to efficiency"""
    df = load_file()
    if df is None:
        return
    required_cols = ['Elims', 'Solo kills', 'Final blows', 'Dmg', 'H', 'Hero']
    if not all(col in df.columns for col in required_cols):
        messagebox.showwarning("Missing data", "Required columns are missing in the file.")
        return

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

    grouped_heroes = df.groupby('Group', observed=False)['Hero'].apply(lambda x: ', '.join(x)).reset_index()
    info = [f"{row['Group']}: {row['Hero']}" for _, row in grouped_heroes.iterrows()]
    set_info(
        "FES is the ratio of eliminations to the sum of solo kills and final blows, shown in relation to efficiency.",
        info[0] if len(info) > 0 else "",
        info[1] if len(info) > 1 else "",
        info[2] if len(info) > 2 else ""
    )
    show_plot(fig)

def set_info(line1, line2, line3, line4):
    """sets the text in the information fields"""
    info0.config(text=line1)
    info1.config(text=line2)
    info2.config(text=line3)
    info3.config(text=line4)

# --- gui ---

root = tk.Tk()
root.title("Game Data Analysis")

# file loading
tk.Label(root, text="File name:").grid(row=0, column=0, padx=10, pady=0, sticky="w")
file_entry = tk.Entry(root)
file_entry.insert(0, "data.txt")
file_entry.grid(row=1, column=0, padx=10, pady=0, sticky="w")

# analysis buttons
tk.Button(root, text="Objective Clusters", command=cluster_analysis).grid(row=4, column=0, padx=10, pady=10, sticky="w")
tk.Button(root, text="Support Efficiency", command=support_efficiency).grid(row=5, column=0, padx=10, pady=10, sticky="w")
tk.Button(root, text="FES vs Efficiency", command=fes_vs_efficiency).grid(row=6, column=0, padx=10, pady=10, sticky="w")
tk.Button(root, text="Close", command=close_app).grid(row=13, column=0, padx=10, pady=10, sticky="w")

# plot preview
preview_frame = tk.Frame(root, width=1140, height=600)
preview_frame.grid(row=0, column=1, rowspan=14, padx=10, pady=10)
preview_frame.grid_propagate(False)

# information fields positions
info0 = tk.Label(root, wraplength=260, justify="left", text=" ")
info0.grid(row=9, column=0, padx=10, pady=0, sticky="w")
info1 = tk.Label(root, wraplength=260, justify="left", text=" ")
info1.grid(row=10, column=0, padx=10, pady=0, sticky="w")
info2 = tk.Label(root, wraplength=260, justify="left", text=" ")
info2.grid(row=11, column=0, padx=10, pady=0, sticky="w")
info3 = tk.Label(root, wraplength=260, justify="left", text=" ")
info3.grid(row=12, column=0, padx=10, pady=0, sticky="w")

root.mainloop()
