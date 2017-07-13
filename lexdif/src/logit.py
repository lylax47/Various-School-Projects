import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import regex as re

def import_data(path):
    df = pd.read_csv(path, delimiter=',')
    return df

def norm(var):
    var_norm = (var - np.amin(var))/(np.amax(var) - np.amin(var))
    return var_norm

def print_res (res_list):
    with open('results.txt', 'w') as text:
        text.write(str(res_list)    )


def regression(title,name_x,name_y,x,y):
    model = sm.OLS(y,x)
    result = model.fit()
    fig, ax = plt.subplots()
    fig = sm.graphics.plot_fit(result, 0, ax=ax)
    ax.set_ylabel(name_y)
    ax.set_xlabel(name_x)
    ax.set_title(title)
    plt.show()
    summ = result.summary()
    return summ

res_list = []
df = import_data('/home/john/PycharmProjects/lexdif/src/data.csv')
col_1 = 1
col_2 = 6
while col_1 < 6:
    y = df[df.columns[col_2]]
    x = df[df.columns[col_1]]
    name_x = df.columns[col_1]
    name_y = df.columns[col_2]
    x = norm(x)
    y = norm(y)
    title = re.sub('_ru', '', name_x)
    res_list.append(regression(title,name_x,name_y,x,y))
    col_1 += 1
    col_2 += 1
print_res(res_list)
