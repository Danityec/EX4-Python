import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.inspection import permutation_importance


def pair_plot_species(data):
    sns.pairplot(data, hue='species')
    plt.show()


def categorical_to_numerical(data):
    island_list = ['Torgersen', 'Biscoe', 'Dream']
    data['island_bin'] = pd.Categorical(data.island, ordered=False, categories=island_list).codes

    sex_list = ['Male', 'Female']
    data['sex_bin'] = pd.Categorical(data.sex, ordered=False, categories=sex_list).codes

    return data.drop(['island', 'sex'], axis=1)


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
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])

    if prob:
        z = z[:, 1] - z[:, 0]
    else:
        colors = "Set1"
        z = np.argmax(z, axis=1)

    z = z.reshape(xx.shape)

    CP = plt.contourf(xx, yy, z, cmap=colors, alpha=0.5)
    plt.colorbar()
    if not prob:
        plt.clim(0, len(clf.classes_) + 3)

    # new_data = pd.DataFrame(columns=['bill_length_mm', 'bill_depth_mm', 'species'])
    # new_data.loc[0] = ([] + [] + ['Adelie'])
    # new_data.loc[1] = ([] + [] + ['Chinstrap'])
    # new_data.loc[2] = ([] + [] + ['Gentoo'])

    plt.xlabel('bill_length_mm')
    plt.ylabel('bill_depth_mm')
    # plt.legend(['Adelie', 'Chinstrap', 'Gentoo'])
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    plt.show()


def bayes_plot_with_failed_scatter(df):
    y = df[df.columns[2]]
    x = df.drop(df.columns[2], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    clf = GaussianNB()
    clf = clf.fit(x_train, y_train)

    prob = len(clf.classes_) == 2

    x_min, x_max = x.loc[:, 'bill_length_mm'].min() - 1, x.loc[:, 'bill_length_mm'].max() + 1
    y_min, y_max = x.loc[:, 'bill_depth_mm'].min() - 1, x.loc[:, 'bill_depth_mm'].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # x_min, x_max = x.loc[:, col1].min() - 1, x.loc[:, col1].max() + 1
    # y_min, y_max = x.loc[:, col2].min() - 1, x.loc[:, col2].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

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

    y_pred_full = clf.predict(x)
    y_pred_series = pd.Series(y_pred_full)
    failed_data = []
    for i in range(len(y_pred_series)):
        if y_pred_series[y_pred_series.index[i]] != y[y.index[i]]:
            failed_data.append(i)

    new_data = pd.DataFrame(columns=['bill_length_mm', 'bill_depth_mm', 'species'])
    i = 0
    for f in failed_data:
        new_data.loc[i] = ([x.loc[f]['bill_length_mm']] + [x.loc[f]['bill_depth_mm']] + [y_pred_series.loc[f]])
        i += 1

    hue_order = clf.classes_
    new_data = new_data.sort_values('species')
    new_data2 = new_data.reset_index(drop=True)
    sns.scatterplot(data=new_data2, x='bill_length_mm', y='bill_depth_mm', hue=new_data['species'], hue_order=hue_order, palette=colors)
    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    plt.show()


def bayes_classification_report(df):
    y = df[df.columns[2]]
    x = df.drop(df.columns[2], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    clf = GaussianNB()
    clf = clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    print(metrics.classification_report(y_test, y_pred))


def clean_data(df):
    df['bill_length_mm'].fillna(round((df['bill_length_mm'].mean()), 2), inplace=True)
    df['bill_depth_mm'].fillna(round((df['bill_depth_mm'].mean()), 2), inplace=True)
    df['flipper_length_mm'].fillna(round((df['flipper_length_mm'].mean()), 2), inplace=True)
    df['body_mass_g'].fillna(round((df['body_mass_g'].mean()), 2), inplace=True)
    df['island'].fillna((df['island'].mode()[0]), inplace=True)
    df['sex'].fillna((df['sex'].mode()[0]), inplace=True)
    df['species'].fillna((df['species'].mode()[0]), inplace=True)
    df = categorical_to_numerical(df)

    return df


if __name__ == '__main__':
    penguins = pd.read_csv("penguins.csv")
    penguins = clean_data(penguins)

    # 1.1:
    # pair_plot_species(penguins.drop(['sex_bin', 'island_bin'], axis=1))

    # 1.2:
    x_penguins = penguins.drop(['species', 'island_bin', 'sex_bin', 'flipper_length_mm', 'body_mass_g'], axis=1)
    y_penguins = penguins['species']

    gaussian_naive_bayes(x_penguins, y_penguins)

    # 1.3:
    bayes_plot(pd.concat([x_penguins, y_penguins], axis=1))

    # 1.4:
    bayes_plot_with_failed_scatter(pd.concat([x_penguins, y_penguins], axis=1))

    # 1.5:
    # bayes_classification_report(pd.concat([x_penguins, y_penguins], axis=1))
