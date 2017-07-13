import regex as re
import pandas as pd
import numpy as np
import nltk
import csv
import itertools as it

def imp_csv ():
    df = pd.read_csv('info.csv')
    df_t = df[['ID', 'Правдивый текст']]
    df_f = df[['ID', 'Ложный текст']]
    df_t['Truth'] = list((0,0)*57)
    df_f['Truth'] = list((1,1)*57)
    df_f.columns = ['ID', 'Text', 'Truth']
    df_t.columns = ['ID', 'Text', 'Truth']
    df2 = pd.concat([df_t, df_f], axis=0)
    df2 = df2.dropna()
    return df2

def tot_wd_avg(lang):
    word_leng = {}
    TTR = {}
    hapax = {}
    yules = {}
    for index, row in lang.iterrows():
        sent_leng_list = []
        words = {}
        sents = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', row['Text'])
        for sent in sents:
            sent_tks = nltk.tokenize.word_tokenize(sent.lower())
            sent_leng = len(sent_tks)
            sent_leng_list.append(sent_leng)
            for word in sent_tks:
                if word in words.keys():
                    words[word] += 1
                else:
                    words[word] = 1
        tot_len = 0
        for length in sent_leng_list:
            tot_len += length
        avg_len = tot_len/len(sent_leng_list)
        lang.ix[index, 'Avg_sent'] = avg_len
        total = 0
        word_tot = 0
        sing = 0
        for word,count in words.items():
            total += count
            word_tot += len(word)*count
            if count == 1:
                sing += 1
        ttr = len(words)/total
        word_avg = word_tot/total
        hapax_val = sing/len(words)
        lang.ix[index, 'TTR'] = ttr
        lang.ix[index, 'Avg_word'] = word_avg
        lang.ix[index, 'Hapax'] = hapax_val
        m1 = len(words)
        m2 = sum([len(list(g))*(freq**2) for freq,g in it.groupby(sorted(words.values()))])
        yulesk = 10000*(m2 - m1)/(m1 * m1)
        try:
            lang.ix[index, 'Yules'] = yulesk
        except ZeroDivisionError:
            lang.ix[index, 'Yules'] = yulesk
    return (lang)


def count_user_avgs(data):
    avgs = tot_wd_avg(data)
    return avgs


def write_csv(final):
    final.to_csv('new_info.csv')


data = imp_csv()
avgs = count_user_avgs(data)
print(avgs)
write_csv(avgs)
