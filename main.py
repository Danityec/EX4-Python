import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn as skl
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


def tests(data):
    pass
    # grid = sns.FacetGrid(data, row="sex", col="species", margin_titles=True)
    # grid.map(plt.hist, "bill_length_mm")
    #
    # grid = sns.FacetGrid(data, row="island_bin", col="species_bin", margin_titles=True)
    # grid.map(plt.hist, "sex_bin")
    # plt.show()


def categorical_to_numerical(data):
    island_list = ['Torgersen', 'Biscoe', 'Dream']
    data['island_bin'] = pd.Categorical(data.island, ordered=False, categories=island_list).codes

    species_list = ['Adelie', 'Chinstrap', 'Gentoo']
    data['species_bin'] = pd.Categorical(data.species, ordered=False, categories=species_list).codes

    sex_list = ['Male', 'Female']
    data['sex_bin'] = pd.Categorical(data.sex, ordered=False, categories=sex_list).codes


def modified_heatmap(data):
    plt.subplots(figsize=(8, 8))
    sns.heatmap(data.corr(), annot=True, fmt="f")
    plt.show()


def pair_plot_sex(data):
    sns.pairplot(data, hue='sex')
    plt.show()


def pair_plot_species(data):
    sns.pairplot(data, hue='species')
    plt.show()


def pair_plot_island(data):
    sns.pairplot(data, hue='island')
    plt.show()


if __name__ == '__main__':
    data = pd.read_csv("penguins.csv")
    data = data.dropna()

    island_list = ['Torgersen', 'Biscoe', 'Dream']
    data['island_bin'] = pd.Categorical(data.island, ordered=False, categories=island_list).codes
    sex_list = ['Male', 'Female']
    data['sex_bin'] = pd.Categorical(data.sex, ordered=False, categories=sex_list).codes

    data = data.drop(['island', 'sex'], axis=1)

    # bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex_bin, island_bin

    x_penguins = data.drop(['species', 'island_bin', 'sex_bin', 'body_mass_g'], axis=1)
    y_penguins = data['species']

    x_train, x_test, y_train, y_test = train_test_split(x_penguins, y_penguins, test_size=0.2, random_state=1)

    model = GaussianNB()
    model.fit(x_train, y_train)
    y_model = model.predict(x_test)
    y_pred = pd.Series(y_model, name='prediction')
    predicted = pd.concat([x_test.reset_index(), y_test.reset_index(), y_pred], axis=1)
    # print(predicted)

    print(metrics.accuracy_score(y_test, y_model))
