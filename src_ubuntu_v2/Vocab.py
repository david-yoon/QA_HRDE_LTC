# coding: utf-8

import numpy as np
import operator

class Vocab:
    
    def __init__(self, dic):
        
        self.dic = dic
        self.index = []
        self._create_index()
    
    def _create_index(self):
        
        sorted_voca = sorted(self.dic.items(), key=operator.itemgetter(1))
        for word, num in sorted_voca:
            self.index.append( word )
            
    def find_index(self, word):
        if self.dic.has_key(word):
            return self.dic[word]
        else:
            return self.dic['_UNK_']  
            
    def index2sent(self, index):
        return ' '.join( [ self.index[int(x)] for x in index ] )
    
    
    def word2index(self, word):
        return " ".join( [ str(self.find_index(x)) for x in word ] )
    

    # 문장을 넣으면 list 로 index 를 return
    def __call__(self, line):        
        """
        if type(line) is np.ndarray:
            return " ".join([self.index2word[word] for word in line])
        
        if type(line) is list:
            if len(line) > 0:
                if line[0] is int:
                    return " ".join([self.index2word[word] for word in line])
            indices = np.zeros(len(line), dtype=np.int32)
        else:
            line = line.split(" ")
            indices = np.zeros(len(line), dtype=np.int32)
        
        for i, word in enumerate(line):
            indices[i] = self.word2index.get(word, self.unknown)
            
        return indices
        """
        
    
    @property
    def size(self):
        return len(self.index2word)
    
    def __len__(self):
        return len(self.index2word)  
    
    


# In[ ]:




# In[ ]:



