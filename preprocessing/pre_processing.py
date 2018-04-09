
# coding: utf-8
'''
what:   generate dictionary from corpus 
        apply mincut frequency
        store result as file        
'''
from __future__ import division

from nltk.tokenize import word_tokenize
import io
import pickle
import csv
import sys
from collections import defaultdict
import argparse

import os
from nltk.tag import StanfordNERTagger

STANFORDTOOLS_DIR = '../stanford-ner-2016-10-31'
NER_PROCESSED = '_ner'
DATA_DIR = '../data/UbuntuDialogs_data/'
    

class PRE_PROCESSING:

    def __init__(self):
        #os.environ['CLASSPATH'] = STANFORDTOOLS_DIR
        #os.environ['STANFORD_MODELS'] = STANFORDTOOLS_DIR + '/classifiers'
        #os.environ['JAVA_OPTIONS'] = '-Xmx8192m'
        #self.st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz')

        self.data = []
        self.frequency_ori = defaultdict(int)
        self.target_file = ''
        self.st = None
        self.min_freq = 0
        
        
    def _data_read(self):

        print 'data read : ' + DATA_DIR + self.target_file

        f = open(DATA_DIR + self.target_file, 'r')
        csvReader = csv.reader(f)
        for row in csvReader:
            self.data.append(row)
        f.close()

        
    def _count_word(self):

        self.frequency_ori = defaultdict(int)
        
        # data[0] -> title information -> should be excluded 
        for row in self.data[1:]:

            # train.csv --> "message", "response", "label"
            for i in xrange(2):
                for token in row[i].split(" "):
                    self.frequency_ori[token] += 1


    def _apply_mincut(self, min_freq):
        
        self.min_freq = min_freq

        print 'apply minCut and re-generate minCutDic'
        mincut_dic = dict(filter(lambda (a, b) : b > self.min_freq, self.frequency_ori.items()))

        print 'minFreq = ' + str(self.min_freq)
        print 'original dic size = ' + str(len(self.frequency_ori))
        print 'original dic word freq = ' + str(sum(self.frequency_ori.values()))
        print 'minCut dic size = ' + str(len(mincut_dic))
        print 'minCut dic word freq = ' + str(sum(mincut_dic.values()))

        coverage = sum(mincut_dic.values()) / sum(self.frequency_ori.values())
        print 'coverage = ' + str(coverage)

        return mincut_dic


    def _save_dic(self, dic_name, mincut_dic):

        print 'write dic as File'

        min_dic_file = DATA_DIR + dic_name

        outFile = open(DATA_DIR + 'dic_ori_'+ self.target_file +'_'+ str(self.min_freq), 'wb')
        outFile2 = open(min_dic_file, 'wb')
        pickle.dump(self.frequency_ori, outFile)
        pickle.dump(mincut_dic, outFile2)

        outFile.close()
        outFile2.close()

        print 'complete'
        print 'min dic file name = ' + min_dic_file


    def _load_dic(self, dic_name):
        f = open(DATA_DIR + dic_name, 'rb')
        mincut_dic = pickle.load(f)
        f.close()
        return mincut_dic


    def _replace_unk(self, dic, target_file, postfix):

        # # Data Convert Useing MinCut Dic
        # replace removed word as **unk**

        _data = []

        f = open(DATA_DIR + target_file, 'r')
        csvReader = csv.reader(f)
        for row in csvReader:
            _data.append(row)
        f.close()


        # train.csv case
        if (target_file == 'train.csv'):
            print 'process: train file'

            f = open(DATA_DIR + target_file + postfix, 'w')
            csvWriter = csv.writer(f)

            for row in _data[1:]:

                tmpSen = ["", ""]
                for i in xrange(2):

                    for token in row[i].split(" "):
                        if (dic.has_key(token)) :
                            tmpSen[i] += token
                            tmpSen[i] += " "
                        else:
                            tmpSen[i] += "_UNK_"
                            tmpSen[i] += " "

                csvWriter.writerow([tmpSen[0].strip(), tmpSen[1].strip(), row[2]])
            f.close()            


        # valid/test.csv case
        else:
            print 'process valid/test file'

            f = open(DATA_DIR + target_file + postfix, 'w')
            csvWriter = csv.writer(f)

            for row in _data[1:]:

                tmpSen = ["", "", "", "", "", "", "", "", "", "", ""]

                for i in xrange(11):

                    for token in row[i].split(" "):
                        if (dic.has_key(token)) :
                            tmpSen[i] += token
                            tmpSen[i] += " "
                        else:
                            tmpSen[i] += "_UNK_"
                            tmpSen[i] += " "

                csvWriter.writerow([tmpSen[0].strip(), tmpSen[1].strip(), tmpSen[2].strip(), tmpSen[3].strip(), tmpSen[4].strip(), 
                                    tmpSen[5].strip(), tmpSen[6].strip(), tmpSen[7].strip(), tmpSen[8].strip(), tmpSen[9].strip(), tmpSen[10].strip()])

            f.close()         


def main(file_name):

    mincut_dic = None
    dic_name = file_name
        
    if not os.path.exists(DATA_DIR + dic_name) :
        _data_read()
        _count_word()
        mincut_dic = _apply_mincut()
        _save_dic(dic_name, mincut_dic)
    
    mincut_dic = _load_dic(dic_name)
    _replace_unk(mincut_dic, 'train.csv', '_processed')
    _replace_unk(mincut_dic, 'valid.csv', '_processed')
    _replace_unk(mincut_dic, 'test.csv', '_processed')

    
if __name__ == '__main__':
    
    p = argparse.ArgumentParser()
    p.add_argument('--dic_name', type=str, default="dic_mincut_15")
    args = p.parse_args()
    
        
    main(file_name=args.dic_name)
    
    
    