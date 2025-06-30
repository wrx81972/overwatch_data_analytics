import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import mplcursors
import tkinter as tk
import seaborn as sns

matplotlib.use('TkAgg')
pd.set_option('display.max_colwidth', None)


def wczytaj_plik():
    plik_txt = pole.get()

    try:
        df = pd.read_csv(plik_txt, sep='\t', encoding='cp1250')
        df['Win rate'] = pd.to_numeric(df['Win rate'], errors='coerce')
        return df
    except Exception as e:
        print(e)


def pokaz(wykres):
    for widget in podglad.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(wykres, master=podglad)
    canvas.draw()
    canvas.get_tk_widget().pack()


def koniec():
    root.quit()


def pokaz_kursor(osi, df):
    kursor = mplcursors.cursor(osi, hover=True)

    @kursor.connect("add")
    def on_add(sel):
        index = sel.index
        sel.annotation.set_text(df.iloc[index]['Hero'])


def klastry(df):
    potrzebne = ['Obj kills', 'OV s']
    if all(col in df.columns for col in potrzebne):
        wybrane = df[potrzebne]
        standaryzuj = StandardScaler()
        ustandaryzowane = standaryzuj.fit_transform(wybrane)
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['Group'] = kmeans.fit_predict(ustandaryzowane)

        ety0.config(
            text="This figure focuses on data related to objective time per character. It uses KMeans algorithm to "
                 "cut existing data into 3 clusters based on similair relation between kills on objective to time "
                 "spent on objective per character. ")
        ety1.config(text="Clusters mostly overlap with heroes roles.")
        ety2.config(text="Please hover points to see heroes' names.")
        ety3.config(text="")

        df['Color'] = df['Group'].astype('int')
        df['Group'] = df['Group'].astype('object')
        df.loc[df['Group'] == 0, 'Group'] = 'Support (low Elims)'
        df.loc[df['Group'] == 1, 'Group'] = 'Tank'
        df.loc[df['Group'] == 2, 'Group'] = 'DPS (high Elims)'

        kolory = {'Support (low Elims)': 0, 'Tank': 1, 'DPS (high Elims)': 2}
        df['Color'] = df['Group'].map(kolory)

        wykres = plt.figure(figsize=(10, 6))
        osi = plt.scatter(df['Obj kills'], df['OV s'], c=df['Color'], cmap='viridis', s=100)
        plt.title('Elims on objective vs. overall objective time [s]')
        plt.xlabel('Eliminations')
        plt.ylabel('Time [s]')
        plt.legend(handles=[
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=osi.cmap(osi.norm(i)), markersize=10)
            for i in range(3)],
            labels=kolory.keys(), title='Groups (Clusters)')
        pokaz_kursor(osi, df)
        pokaz(wykres)

    else:
        print("Brak wymaganych kolumn w danych.")


def wybitnosc_supportow(df):
    df['KDAH'] = 0.0
    df.loc[df['Role'] == 'Support', 'KDAH'] = (df['KDA'] * df['H'] / 1000).astype(float)
    df_support = df[df['Role'] == 'Support']

    wykres, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(df_support['Win rate'], df_support['KDAH'], s=100, c='blue')
    sns.regplot(x='Win rate', y='KDAH', data=df_support, scatter_kws={'s': 100, 'color': 'blue'},
                line_kws={'color': 'red'}, ax=ax)
    ax.set_title('KDAH vs. Win rate')
    ax.set_xlabel('Win rate [%]')
    ax.set_ylabel('KDAH/1000')

    ety0.config(text="KDAH is a new parameter combining KDA and H (healing). The figure shows efficiency correlated "
                     "with win rate for the characters with healing abilities. ")
    ety1.config(text="Please hover points to see heroes' names.")
    ety2.config(text="")
    ety3.config(text="")

    kursor = mplcursors.cursor(sc, hover=True)

    @kursor.connect("add")
    def on_add(sel):
        x_hover = sel.target[0]
        y_hover = sel.target[1]

        distances = np.sqrt((df_support['Win rate'] - x_hover) ** 2 + (df_support['KDAH'] - y_hover) ** 2)
        closest_index = distances.idxmin()
        hero_name = df_support.loc[closest_index, 'Hero']
        sel.annotation.set_text(hero_name)

    pokaz(wykres)


def fes_win(df):
    df['FES'] = df['Elims'] / (df['Solo kills'] + df['Final blows'])
    df['Eff'] = ((df['Dmg'] / df['Elims']) + (df['H'] / 1000))/100

    liczba_przedzialow = 3
    przedzialy = np.linspace(df['Eff'].min(), df['Eff'].max(), liczba_przedzialow + 1)

    df['Group'] = pd.cut(df['Eff'], bins=przedzialy,
                         labels=['Low Efficiency', 'Medium Efficiency', 'High Efficiency'])
    wykres = plt.figure(figsize=(10, 6))
    sns.boxplot(x='Group', y='FES', data=df)

    plt.title('FES vs efficiency group')
    plt.xlabel('Group')
    plt.ylabel('FES')

    grouped_heroes = df.groupby('Group', observed=False)['Hero'].apply(lambda x: ', '.join(x)).reset_index()
    grouped_heroes['Label'] = grouped_heroes.apply(lambda row: f"{row['Group']}: {row['Hero']}", axis=1)

    ety0.config(text="FES is Elims divided by Solo kills + Final blows. It is shown in relation to the efficiency as "
                     "of kill per DMG dealt.")
    ety1.config(text=grouped_heroes['Label'].values[0])
    ety2.config(text=grouped_heroes['Label'].values[1])
    ety3.config(text=grouped_heroes['Label'].values[2])

    pokaz(wykres)


root = tk.Tk()
root.title("analiza, janik p.")

etykieta_label = tk.Label(root, text="File name: ")
etykieta_label.grid(row=0, column=0, padx=10, pady=0, sticky="w")
pole = tk.Entry(root)
pole.insert(0, "dane.txt")
pole.grid(row=1, column=0, padx=10, pady=0, sticky="w")

an1_przycisk = tk.Button(root, text="Objective", command=lambda: klastry(wczytaj_plik()))
an1_przycisk.grid(row=4, column=0, padx=10, pady=10, sticky="w")

an2_przycisk = tk.Button(root, text="KDAH x wins", command=lambda: wybitnosc_supportow(wczytaj_plik()))
an2_przycisk.grid(row=5, column=0, padx=10, pady=10, sticky="w")

an3_przycisk = tk.Button(root, text="FES x wins", command=lambda: fes_win(wczytaj_plik()))
an3_przycisk.grid(row=6, column=0, padx=10, pady=10, sticky="w")

zamknij_przycisk = tk.Button(root, text="Close window", command=koniec)
zamknij_przycisk.grid(row=13, column=0, padx=10, pady=10, sticky="w")

podglad = tk.Frame(root, width=1140, height=600)
podglad.grid(row=0, column=1, rowspan=14, padx=10, pady=10)
podglad.grid_propagate(False)

ety0 = tk.Label(root, wraplength=260, justify="left", text=" ")
ety0.grid(row=9, column=0, padx=10, pady=0, sticky="w")
ety1 = tk.Label(root, wraplength=260, justify="left", text=" ")
ety1.grid(row=10, column=0, padx=10, pady=0, sticky="w")
ety2 = tk.Label(root, wraplength=260, justify="left", text=" ")
ety2.grid(row=11, column=0, padx=10, pady=0, sticky="w")
ety3 = tk.Label(root, wraplength=260, justify="left", text=" ")
ety3.grid(row=12, column=0, padx=10, pady=0, sticky="w")

root.mainloop()
