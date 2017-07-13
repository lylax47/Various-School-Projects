import os
import sys
import pandas as pd
import sklearn.metrics as mt
import matplotlib.pyplot as plt
import json
import regex as re
from nltk.tokenize import sent_tokenize
from pymystem3 import Mystem
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

def pos_bi(text):
    pos_tags = []
    m = Mystem()
    sents = sent_tokenize(text)
    for sent in sents:
        sent_an = []
        analy = m.analyze(sent)
        for x in analy:
            try:
                if 'analysis' in x.keys():
                    tag = x['analysis'][0]['gr']
                    sent_an.append(re.sub(r'[=|,].*', '', tag).lower())
            except IndexError:
                pass
        pos_tags.append(sent_an)
    return pos_bi


def create_corp(texts):
    text_all = []
    i = 0
    for text in texts:
        i += 1
        if i > 500:
            sys.exit()
        # text_all.append(pos_bi(text))
    print('hello')
    sys.exit()


def imp (filenames, root):
    sheets =[]
    for f in filenames:
        if f != 'sex.csv' and f != 'new_info.csv':
            df = pd.read_csv(os.path.join(root, f), index_col=0)
            sheets.append(df)
        elif f == 'sex.csv':
            df2 = pd.read_csv(os.path.join(root, f))
        else:
            df3 = pd.read_csv(os.path.join(root, f))
    comb = pd.concat(sheets, axis=1)
    new_vec = list((1,0)*113)
    comb['Truth'] = new_vec
    comb['Sex'] = ''
    comb['Avg_sent'], comb['TTR'], comb['Avg_word'], comb['Hapax'], comb['Yules'] = 0,0,0,0,0
    lie_corp = df3[df3['Truth'] == 1]
    true_corp = df3[df3['Truth'] == 0]
    lies = create_corp(lie_corp['Text'])
    truths = create_corp(lie_corp['Text'])
    for num in df2['ID']:
        num2 = num - 1
        comb.loc['{0}Л.docx'.format(num), 'Sex'] = df2.ix[num2, 'Пол']
        comb.loc['{0}П.docx'.format(num), 'Sex'] = df2.ix[num2, 'Пол']
    for index, row in df3.iterrows():
        r = int(row['ID'])
        text = str(row['Text'])
        if row['Truth'] == 1:
            comb.loc['{0}Л.docx'.format(r), ['Avg_sent']] = row['Avg_sent']
            comb.loc['{0}Л.docx'.format(r), ['TTR']] = row['TTR']
            comb.loc['{0}Л.docx'.format(r), ['Avg_word']] = row['Avg_word']
            comb.loc['{0}Л.docx'.format(r), ['Hapax']] = row['Hapax']
            comb.loc['{0}Л.docx'.format(r), ['Yules']] = row['Yules']
            comb.loc['{0}Л.docx'.format(r), ['']] = row['Text']
        else:
            comb.loc['{0}П.docx'.format(r), ['Avg_sent']] = row['Avg_sent']
            comb.loc['{0}П.docx'.format(r), ['TTR']] = row['TTR']
            comb.loc['{0}П.docx'.format(r), ['Avg_word']] = row['Avg_word']
            comb.loc['{0}П.docx'.format(r), ['Hapax']] = row['Hapax']
            comb.loc['{0}П.docx'.format(r), ['Yules']] = row['Yules']
            comb.loc['{0}П.docx'.format(r), ['']] = row['Text']
    return comb


def log_reg(df):
    train = df.sample(frac=0.9, random_state=1)
    test = df.loc[~df.index.isin(train.index)]
    xtrain = train[['Личноеместоимение', 'вербальные', 'Предлог', 'Союз', 'Когнитив',
                    'Включение', 'AllPunc', 'мест.-сущ.', 'и', 'добавление',
                    'Avg_sent', 'TTR', 'Avg_word', 'Hapax', 'Yules']]
    xtest = test[['Личноеместоимение', 'вербальные', 'Предлог', 'Союз', 'Когнитив',
                    'Включение', 'AllPunc', 'мест.-сущ.', 'и', 'добавление',
                  'Avg_sent', 'TTR', 'Avg_word', 'Hapax', 'Yules']]
    ytrain = train['Truth']
    ytest = test['Truth']
    reg = linear_model.LogisticRegressionCV()
    reg.fit(xtrain,ytrain)
    preds = reg.predict(xtest)
    print(reg.coef_)
    print(mean_squared_error(preds, ytest))
    print(reg.score(xtest, ytest))
    print(mt.r2_score(ytest, preds))
    print(mt.classification_report(ytest, preds, target_names=['Truth', 'Lie']))
    print(mt.accuracy_score(ytest, preds))
    print(mt.confusion_matrix(ytest, preds))


def log_sex(df):
    mal = df[df['Sex'] == 'муж.']
    fem = df[df['Sex'] == 'жен.']

    print(len(mal))
    print(len(fem))

    train_mal = mal.sample(frac=0.6, random_state=1)
    train_fem = fem.sample(frac=0.6, random_state=1)

    test_mal = df.loc[~mal.index.isin(train_mal.index)]
    test_fem = df.loc[~fem.index.isin(train_fem.index)]

    xtrain_mal = train_mal[['Личноеместоимение', 'вербальные', 'Предлог', 'Союз', 'Когнитив',
                    'Включение', 'AllPunc', 'мест.-сущ.', 'и', 'добавление',
                            'Avg_sent', 'TTR', 'Avg_word', 'Hapax', 'Yules']]
    xtrain_fem = train_fem[['Личноеместоимение', 'вербальные', 'Предлог', 'Союз', 'Когнитив',
                    'Включение', 'AllPunc', 'мест.-сущ.', 'и', 'добавление',
                            'Avg_sent', 'TTR', 'Avg_word', 'Hapax', 'Yules']]
    xtest_mal = test_mal[['Личноеместоимение', 'вербальные', 'Предлог', 'Союз', 'Когнитив',
                    'Включение', 'AllPunc', 'мест.-сущ.', 'и', 'добавление',
                          'Avg_sent', 'TTR', 'Avg_word', 'Hapax', 'Yules']]
    xtest_fem = test_fem[['Личноеместоимение', 'вербальные', 'Предлог', 'Союз', 'Когнитив',
                    'Включение', 'AllPunc', 'мест.-сущ.', 'и', 'добавление',
                          'Avg_sent', 'TTR', 'Avg_word', 'Hapax', 'Yules']]

    ytrain_mal = train_mal['Truth']
    ytrain_fem = train_fem['Truth']
    ytest_mal = test_mal['Truth']
    ytest_fem = test_fem['Truth']

    reg_mal = linear_model.LogisticRegressionCV()
    reg_fem = linear_model.LogisticRegressionCV()

    reg_mal.fit(xtrain_mal, ytrain_mal)
    preds_mal = reg_mal.predict(xtest_mal)

    reg_fem.fit(xtrain_fem, ytrain_fem)
    preds_fem = reg_fem.predict(xtest_fem)

    print(reg_mal.coef_)
    print(mean_squared_error(preds_mal, ytest_mal))
    print(reg_mal.score(xtest_mal, ytest_mal))
    print(mt.r2_score(ytest_mal, preds_mal))
    print(mt.classification_report(ytest_mal, preds_mal, target_names=['Truth', 'Lie']))
    print(mt.confusion_matrix(ytest_mal, preds_mal))

    print(reg_fem.coef_)
    print(mean_squared_error(preds_fem, ytest_fem))
    print(reg_fem.score(xtest_fem, ytest_fem))
    print(mt.r2_score(ytest_fem, preds_fem))
    print(mt.classification_report(ytest_fem, preds_fem, target_names=['Truth', 'Lie']))
    print(mt.confusion_matrix(ytest_fem, preds_fem))


# def plot_func(df):
#     pca = PCA(1)
#     x = df[['Личноеместоимение', 'вербальные', 'Предлог', 'Союз', 'Когнитив',
#            'Включение', 'AllPunc', 'мест.-сущ.', 'и', 'добавление']]
#     x_red = pca.fit_transform(x)
#     train = df.sample(frac=0.7, random_state=1)
#     test = df.loc[~df.index.isin(train.index)]
#     xtrain = train[['Личноеместоимение', 'вербальные', 'Предлог', 'Союз', 'Когнитив',
#                     'Включение', 'AllPunc', 'мест.-сущ.', 'и', 'добавление']]
#     xtest = test[['Личноеместоимение', 'вербальные', 'Предлог', 'Союз', 'Когнитив',
#                     'Включение', 'AllPunc', 'мест.-сущ.', 'и', 'добавление']]
#     ytrain = train['Truth']
#     ytest = test['Truth']
#     plt.scatter(xtest, ytest, color='blue')
#     plt.plot(xtest, reg.predict(xtest), color='red', linewidth=3)
#     plt.xticks(())
#     plt.yticks(())
#     plt.show()


for root, dirs, files in os.walk('russian_deception_bank/tables/'):
    results = imp(files, root)
log_reg(results)
log_sex(results)
# plot_func(results)
