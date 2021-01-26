import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn as skl
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


def bayes_plot(df, model="gnb", spread=30):
    df.dropna()
    colors = 'seismic'
    col1 = df.columns[0]
    col2 = df.columns[1]
    target = df.columns[2]
    sns.scatterplot(data=df, x=col1, y=col2, hue=target)
    plt.show()
    y = df[target]  # Target variable
    X = df.drop(target, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    clf = GaussianNB()
    if (model != "gnb"):
        clf = DecisionTreeClassifier(max_depth=model)
    clf = clf.fit(X_train, y_train)

    # Train Classifer
    prob = len(clf.classes_) == 2

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    print(metrics.classification_report(y_test, y_pred))

    hueorder = clf.classes_

    def numify(val):
        return np.where(clf.classes_ == val)[0]

    Y = y.apply(numify)
    x_min, x_max = X.loc[:, col1].min() - 1, X.loc[:, col1].max() + 1
    y_min, y_max = X.loc[:, col2].min() - 1, X.loc[:, col2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                         np.arange(y_min, y_max, 0.2))

    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    if prob:

        Z = Z[:, 1] - Z[:, 0]
    else:
        colors = "Set1"
        Z = np.argmax(Z, axis=1)

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=colors, alpha=0.5)
    plt.colorbar()
    if not prob:
        plt.clim(0, len(clf.classes_) + 3)
    sns.scatterplot(data=df[::spread], x=col1, y=col2, hue=target, hue_order=hueorder, palette=colors)
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    plt.show()


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

    x_penguins = data.drop(['species', 'island_bin', 'sex_bin', 'bill_depth_mm', 'body_mass_g'], axis=1)
    y_penguins = data['species']

    x_train, x_test, y_train, y_test = train_test_split(x_penguins, y_penguins, test_size=0.2, random_state=1)

    model = GaussianNB()
    model.fit(x_train, y_train)
    y_model = model.predict(x_test)
    y_pred = pd.Series(y_model, name='prediction')
    predicted = pd.concat([x_test.reset_index(), y_test.reset_index(), y_pred], axis=1)
    # print(predicted)

    print(metrics.accuracy_score(y_test, y_model))

    bayes_plot(pd.concat([x_penguins, y_penguins], axis=1), spread=1)
