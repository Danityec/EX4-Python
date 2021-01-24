import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def tests():
    x_index = 0
    y_index = 1

    formatter = plt.FuncFormatter(lambda i, *args: data.species[int(i)])

    plt.figure(figsize=(5, 4))
    plt.scatter(data.data[:, x_index], data.data[:, y_index], c=data.species_bin)
    plt.colorbar(ticks=[0, 1, 2], format=formatter)
    plt.xlabel(data.feature_names[x_index])
    plt.ylabel(data.feature_names[y_index])

    plt.tight_layout()
    plt.show()
    # grid = sns.FacetGrid(data, row="sex", col="species", margin_titles=True)
    # grid.map(plt.hist, "bill_length_mm")

    # grid = sns.FacetGrid(data, row="island_bin", col="species_bin", margin_titles=True)
    # grid.map(plt.hist, "sex_bin")
    plt.show()


def categorical_to_numerical():
    island_list = ['Torgersen', 'Biscoe', 'Dream']
    data['island_bin'] = pd.Categorical(data.island, ordered=False, categories=island_list).codes

    species_list = ['Adelie', 'Chinstrap', 'Gentoo']
    data['species_bin'] = pd.Categorical(data.species, ordered=False, categories=species_list).codes

    sex_list = ['Male', 'Female']
    data['sex_bin'] = pd.Categorical(data.sex, ordered=False, categories=sex_list).codes


def modified_heatmap():
    plt.subplots(figsize=(8, 8))
    sns.heatmap(data.corr(), annot=True, fmt="f")
    plt.show()


def pair_plot_sex():
    sns.pairplot(data, hue='sex')
    plt.show()


def pair_plot_species():
    sns.pairplot(data, hue='species')
    plt.show()


def pair_plot_island():
    sns.pairplot(data, hue='island')
    plt.show()


def pair_plot():
    sns.pairplot(data, hue='island')
    plt.show()


if __name__ == '__main__':
    data = pd.read_csv("penguins.csv")
    # print(data.head())
    # pair_plot_sex()
    # pair_plot_island()
    # pair_plot_species()
    categorical_to_numerical()
    modified_heatmap()
    # tests()
    # pair_plot()
