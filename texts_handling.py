# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 07:35:27 2015

@author: bordingj
"""

import pandas as pd
import numpy as np
import math

def Get_index2char_and_char2index_dicts(Texts_list, unseen_tag, end_of_doc_tag):
    """
    In: 
        Texts_list: iteratble of texts
        unseen_tag (string): tag for an unseen character
        end_of_doc_tag (string): tag for end of document
    Out: 
        index2char (dict): dictionary which maps an integer to a character
        char2index (dict): dictionary which maps a character to an integer
    """
    all_chars =[]
    #iterate through all texts
    for i, text in enumerate(Texts_list):
        #split current texts into a list of chars and append it to all-chars list
        all_chars += list(text)
        #to preserve memory and for better efficiency, 
        #remove all duplicate chars from all chars at every 5th iteration
        if (i%5)==0:
            all_chars = list(set(all_chars))
    all_chars = list(set(all_chars))
    # make char2index and index2char dictionaries
    char2index = dict(zip(all_chars, list(range(len(all_chars)))))
    index2char = dict(zip(list(range(len(all_chars))), all_chars))
    
    # add an index for an unseen char
    idx = len(index2char)
    index2char[idx] = unseen_tag
    char2index[unseen_tag] = idx
    # and index for end of document
    idx += 1
    index2char[idx] = end_of_doc_tag
    char2index[end_of_doc_tag] = idx
    
    return char2index, index2char

def GetCharsFromIndices(indices, index2charDict):
    """
    In:
        indices: an iterable of integers/indices
        index2charDict (dict): dictionary which maps an integer to a character
    Out:
        list of single characters
    """ 
    return list(map(lambda x: index2charDict[x], indices))

def AppendDiseases2Dicts(char2indexDict, index2charDict, Diseases, unknown_dis_tag):
    """
    In:
        char2indexDict (dict): dictionary which maps an integer to a character
        index2charDict (dict): dictionary which maps a character to an integer
        Diseases: iterable of diseases as strings
        unknown_dis_tag (string): tag for an unknown disease
    Out:
        char2indexDict: dictionary appended with diseases mapping to indices
        index2charDict: dictionary appended new indices mapping to diseases
    """
    N = len(char2indexDict)
    if N != len(index2charDict):
        raise ValueError('char2indexDict and index2charDict must have the same lengths')
    
    all_diseases = list(set(Diseases))
    dis2index = dict(zip(all_diseases, list(range(N,N+len(all_diseases)))))
    UpdatedChar2indexDict = char2indexDict.copy()
    UpdatedChar2indexDict.update(dis2index)
    index2dis = dict(zip(list(range(N,N+len(all_diseases))), all_diseases))
    UpdatedIndex2charDict = index2charDict.copy()
    UpdatedIndex2charDict.update(index2dis)
    # add an index for an unknown disease
    idx = len(UpdatedIndex2charDict)
    UpdatedIndex2charDict[idx] = unknown_dis_tag
    UpdatedChar2indexDict[unknown_dis_tag] = idx
    return UpdatedChar2indexDict, UpdatedIndex2charDict
    
def GetIndicesFromChars(chars, char2indexDict, unseen_tag, end_of_doc_tag):
    """
    In:
        chars: iterable of characters
        char2indexDict (dict): dictionary which maps a character to an integer
        unseen_tag (string): tag for an unseen character
        end_of_doc_tag (string): tag for end of document
    Out:
        1D numpy array of int32 with corresponding indices with appended end-of-document tag
    """
    def GetIndexFromChar(char):
        try:
            index = char2indexDict[char]
        except:
            index = char2indexDict[unseen_tag]
        return index
        
    indices = list(map(lambda x: GetIndexFromChar(x), chars))
    indices.append(char2indexDict[end_of_doc_tag])
    return np.array(indices, dtype=np.int32)

def GetIndicesFromCharsAndDiseases(chars, disease, char2indexDict, unseen_tag, 
                                   unknown_dis_tag, end_of_doc_tag):
    """
    In:
        chars: iterable of characters
        disease (string): disease
        char2indexDict (dict): dictionary which maps a character/disease to an integer
        unseen_tag (string): tag for an unseen character
        end_of_doc_tag (string): tag for end of document
    Out:
        2D numpy array of int32 with corresponding indices:
            1.column: indices for characters with appended end-of-document tag
            2.column: indices for disease
    """
    def GetIndexFromChar(char):
        try:
            index = char2indexDict[char]
        except:
            index = char2indexDict[unseen_tag]
        return index
    def GetIndexFromDisease(dis):
        try:
            index = char2indexDict[dis]
        except:
            index = char2indexDict[unknown_dis_tag]
        return index
        
    indices_chars = list(map(lambda x: GetIndexFromChar(x), chars))
    indices_chars.append(char2indexDict[end_of_doc_tag])
    try:
        dis_index = char2indexDict[disease]
    except:
        dis_index = char2indexDict[unknown_dis_tag]
    indices_dis = len(indices_chars)*[dis_index]
    return np.array([indices_chars,indices_dis], dtype=np.int32).T


class SamplesGenerator(object):
    def __init__(self, corpus, texts_colName, disease_colName,
                     unseen_tag='(UNSEEN)', end_of_doc_tag='(THE_END)', 
                    unknown_dis_tag='UNKNOWN_DIS',doc_len_cutoff = 30):
        """
        This is an object which is able to generate 3D samples for the corpus.
        In:
            corpus (DataFrame): texts corpus as pandas DataFrame
            texts_colName: column name for texts
            disease_colName: columns name for diseases
            unseen_tag (string): tag for an unseen character
            end_of_doc_tag (string): tag for end of document
            unknown_dis_tag (string): tag for an unknown disease
        """
        
        self.corpus = corpus
        self.texts_colName = texts_colName
        self.disease_colName = disease_colName
        self.unseen_tag = unseen_tag
        self.end_of_doc_tag = end_of_doc_tag
        self.unknown_dis_tag = unknown_dis_tag
        self.corpus['text_length'] = [len(x) for x in self.corpus['text']]
        
        self.corpus.sort(columns=['text_length'], inplace=True)
        self.corpus = self.corpus[self.corpus['text_length']>30]
        #generate dictionaries
        self.char2index_charOnly, self.index2char_charOnly = Get_index2char_and_char2index_dicts(
                                                            self.corpus[texts_colName],
                                                             unseen_tag, 
                                                             end_of_doc_tag)
        self.char2index_withDis, self.index2char_withDis = AppendDiseases2Dicts(
                                        char2indexDict=self.char2index_charOnly, 
                                         index2charDict=self.index2char_charOnly, 
                                         Diseases=self.corpus[disease_colName], 
                                        unknown_dis_tag=unknown_dis_tag)
        self.max_size = None
        self.big_empty_batch = None       
        self.N = self.corpus.shape[0]
        self.samples_ordered_by_len = []
        for i in range(self.N):
            sample = GetIndicesFromCharsAndDiseases(chars=self.corpus[texts_colName].iloc[i], 
                               disease=self.corpus[disease_colName].iloc[i], 
                                char2indexDict=self.char2index_withDis, 
                                unseen_tag=unseen_tag,
                                 unknown_dis_tag=unknown_dis_tag, 
                                 end_of_doc_tag=end_of_doc_tag)
            self.samples_ordered_by_len.append(sample)
        self.Max_T = self.samples_ordered_by_len[-1].shape[0]
        

        
    def GetNextBatch(self, max_size, epoch):
        """
        Generates a 3D batch of shape (T, min(max_size,idx_start-N), 2),
        where T is the maximum sequential length of the samples in the batch,
        N is the total number of text samples in the corpus
        and idx_start is the current sample starting index 
        """
        n_batches = math.ceil(self.N/max_size)
        k = epoch%n_batches
        idx_start = k*max_size
        idx_end = min(idx_start+max_size,self.N)
        if self.big_empty_batch is None or max_size != self.max_size or k==0:
            self.max_size = max_size
            single_step_unknown = np.atleast_2d(
                        np.array([self.char2index_withDis[self.end_of_doc_tag],
                                  self.char2index_withDis[self.unknown_dis_tag]],
                                   dtype=np.int32))
            D = single_step_unknown.shape[1]
            single_step_unknown = np.repeat(single_step_unknown, max_size, axis=0)
            self.big_empty_batch = np.repeat(single_step_unknown, 
                                             self.Max_T, axis=0).reshape(self.Max_T,max_size,D,
                                                                    order='C')
        
        
        
        samples = self.samples_ordered_by_len[idx_start:idx_end]
        batch = self.big_empty_batch
        for i,sample in enumerate(samples):
            batch[np.arange(sample.shape[0]),i,:] = sample
        
        max_steps = self.samples_ordered_by_len[(idx_end-1)].shape[0]
        batch = batch[:max_steps,:(i+1),:]
        return np.ascontiguousarray(batch)
    
    def GetCharsFromIndices(self, indices):
        """
        In: 1D iterable of indices for chars
        Out: Text
        """
        return ''.join(GetCharsFromIndices(indices, self.index2char_charOnly))
    
        
#%% Example Usage
#read in corpus
# corpus = pd.read_pickle('/home/bordingj/data/findZebra_corpus.pkl')
                                                 
#make a SamplesGenerator object
#this will take some time and consume alot of memory but give better performance when generating batches
# BatchGenerator = SamplesGenerator(corpus=corpus,
  #                                texts_colName='text', 
   #                               disease_colName='disease_name')
#Generate Batches
# def do_some_iterations():
#    for i in range(500):
 #       Batch = BatchGenerator.GetNextBatch(100,i)
                                
# Get Back Text
# text = BatchGenerator.GetCharsFromIndices(Batch[:,0,0])