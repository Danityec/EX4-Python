import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd


def new_colums_sexANDspecies():

    df1 = penguins['species']
    df2 = penguins['sex']
    penguins.fillna(penguins.mean(), inplace=True)

    penguins["_class"] = (df1 + ' ' + df2)
    #print(penguins)
    return penguins

def pair_plotClass(data):
    sns.pairplot(data, hue='_class')
    plt.show()


def categorical_to_numerical(data):

    island_list = ['Torgersen', 'Biscoe', 'Dream']
    data['island_bin'] = pd.Categorical(data.island, ordered=False, categories=island_list).codes

    return data.drop(['species', 'sex'], axis=1)


def gaussian_naive_bayes(x_data, y_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1)

    gnb_model = GaussianNB()
    gnb_model.fit(x_train, y_train)
    y_model = gnb_model.predict(x_test)

    print(metrics.accuracy_score(y_test, y_model))

def bayes_plot(df):
    colors = 'seismic'
    col1 = df.columns[0]
    col2 = df.columns[1]

    y = df[df.columns[2]]
    x = df.drop(df.columns[2], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    clf = GaussianNB()
    clf = clf.fit(x_train, y_train)

    prob = len(clf.classes_) == 2

    x_min, x_max = x.loc[:, col1].min() - 1, x.loc[:, col1].max() + 1
    y_min, y_max = x.loc[:, col2].min() - 1, x.loc[:, col2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))

    z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])

    if prob:
        z = z[:, 1] - z[:, 0]
    else:
        colors = "Set1"
        z = np.argmax(z, axis=1)

    z = z.reshape(xx.shape)

    plt.contourf(xx, yy, z, cmap=colors, alpha=0.5)
    plt.colorbar()
    if not prob:
        plt.clim(0, len(clf.classes_) + 3)

    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    plt.show()


if __name__ == '__main__':
    penguins = pd.read_csv("penguins.csv")
    penguins = penguins.dropna()

    # 2.0:
    penguins = new_colums_sexANDspecies()

    # 2.1:
    pair_plotClass(penguins)

    # 2.2:
    # species,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex
    penguins = categorical_to_numerical(penguins)
    x_penguins = penguins.drop(['_class', 'sex', 'species', 'bill_depth_mm', 'body_mass_g'], axis=1)
    y_penguins = penguins['_class']
    gaussian_naive_bayes(x_penguins, y_penguins)

    # 2.3:
    # bayes_plot(pd.concat([x_penguins, y_penguins], axis=1))

    # 2.4:
    # bayes_plot_with_failed_scatter(pd.concat([x_penguins, y_penguins], axis=1))

    # 2.5:
    # bayes_classification_report(pd.concat([x_penguins, y_penguins], axis=1))
