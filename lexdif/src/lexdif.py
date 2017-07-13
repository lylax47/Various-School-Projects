import regex as re
import os
import nltk
import csv
import itertools as it

def imp_csv ():
    f = {}
    eng_data = {}
    rus_data = {}
    for root,dirs,files in os.walk('.'):
        for x in files:
            file = os.path.join(root, x)
            f.update({x:file})
    for name,fl in f.items():
        name = re.sub('(_tweets_rus.csv|_tweets_eng.csv)', '', name)
        with open(fl, 'r', encoding = 'utf8', errors='ignore') as txt:
            if re.search('rus', fl):
                rus_data.update({name:txt.read()})
            elif re.search('eng', fl):
                eng_data.update({name:txt.read()})
    return (rus_data, eng_data)

def rt_swap(lang):
    for user, text in lang.items():
        lang[user] = text.split('\n')
        lang[user] = [x for x in lang[user] if x]
        new_sents = []
        for sent in lang[user]:
            re.sub('\s+', ' ', sent)
            re.sub('_+', '', sent)
            if not re.match('("?RT|"|\s)', sent):
                new_sents.append(sent)
        lang[user] = new_sents
    return lang

def clean_texts(rus, eng):
    rus = rt_swap(rus)
    eng = rt_swap(eng)
    return (rus, eng)

def tot_wd_avg(lang):
    sent_lengs = {}
    word_leng = {}
    TTR = {}
    hapax = {}
    yules = {}
    for user,text in lang.items():
        sent_leng_list = []
        words = {}
        for sent in text:
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
        sent_lengs.update({user:avg_len})
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
        TTR.update({user:ttr})
        word_leng.update({user:word_avg})
        hapax.update({user:hapax_val})
        m1 = len(words)
        m2 = sum([len(list(g))*(freq**2) for freq,g in it.groupby(sorted(words.values()))])
        yulesk = 10000*(m2 - m1)/(m1 * m1)
        try:
            yules.update({user:yulesk})
        except ZeroDivisionError:
            yules.update({user:yulesk})
    return (sent_lengs, word_leng, TTR, hapax, yules)

def count_user_avgs(rus, eng):
    rus = tot_wd_avg(rus)
    eng = tot_wd_avg(eng)
    return (rus, eng)

def orgi(rus, eng):
    final = {}
    sent_lengs_ru = rus[0]
    word_lengs_ru = rus[1]
    TTR_ru = rus[2]
    hapax_ru = rus[3]
    yules_ru = rus[4]
    sent_lengs_en = eng[0]
    word_lengs_en = eng[1]
    TTR_en = eng[2]
    hapax_en = eng[3]
    yules_en = eng[4]
    namel = sorted(sent_lengs_ru)
    for name in namel:
        final.update({name:[sent_lengs_ru[name], word_lengs_ru[name], TTR_ru[name], hapax_ru[name], yules_ru[name],
                            sent_lengs_en[name], word_lengs_en[name], TTR_en[name], hapax_en[name], yules_en[name]]})
    return final


def organize(rus, eng):
    final = orgi(rus, eng)
    return final

def write_csv(final):
    with open('data.csv', 'w') as csvfile:
        wr = csv.writer(csvfile, delimiter = ',')
        wr.writerow(['User','sent_leng_ru', 'word_leng_ru', 'TTR_ru', 'Hapax_ru', 'Yules_K_ru',
                    'sent_leng_en', 'word_leng_en', 'TTR_en', 'Hapax_en', "Yules_K_en"])
        for user, values in final.items():
            wr.writerow([user, values[0], values[1], values[2], values[3], values[4],
                          values[5], values[6], values[7], values[8], values[9]])


data = imp_csv()
data = clean_texts(data[0],data[1])
avgs = count_user_avgs(data[0], data[1])
org = organize(avgs[0], avgs[1])
write_csv(org)
