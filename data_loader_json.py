# -*- coding: utf-8 -*-
# 2019-01 
# written by Xiaohui Zhao
# xiaohui.zhao@accenture.com
from os import walk
from os.path import isfile, join
import csv, re, random, json

import numpy as np
import tensorflow as tf
from _operator import index

DEBUG = False

import unicodedata
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass 
    return False

class DataLoader():
    """
    training_grid_tables
    """
    def __init__(self, params, for_train=True, load_dictionary=False, data_split=0.75):
        ## 0> parameters to be tuned
        self.load_dictionary = load_dictionary # load dictionary from file rather than start from empty        
        self.dict_path = params.load_dict_from_path if self.load_dictionary else params.dict_path
        if self.load_dictionary:
            self.dictionary = np.load(self.dict_path + '_dictionary.npy').item()
            self.word_to_index = np.load(self.dict_path + '_word_to_index.npy').item()
            self.index_to_word = np.load(self.dict_path + '_index_to_word.npy').item()
        else:
            self.dictionary = {'<dontcare>':0, '<unknown>':0, '<name>':0} # word/counts. to be updated in self.load_data() and self._update_docs_dictionary()
            self.word_to_index = {}
            self.index_to_word = {}
        self.classes = ['DontCare', 'VendorName', 'VendorTaxID', 'InvoiceDate', 'InvoiceNumber', 'ExpenseAmount', 'BaseAmount', 'TaxAmount', 'TaxRate']

        self.data_split = data_split # split data to training/validation, 0 for all for validation
        self.data_mode = 2 # 0 to consider key and value as two different class, 1 the same class, 2 only value considered
        self.remove_lowfreq_words = False # remove low frequency words when set as True
        
        self.rows = 32 # to be updated in self._update_training_rows_cols()
        self.cols = 32 # to be updated in self._update_training_rows_cols()
        self.encoding_factor = 16 # ensures the size (rows/cols) of grid table compat with the network
        
        self.num_classes = len(self.classes) 
        self.batch_size = params.batch_size if hasattr(params, 'batch_size') else 0        
        
        # TBD: build a special cared dictionary
        self.special_dict = {'some_word': '<dontcare>'} # map texts to specific tokens        
        
        ## 1> load words and their location/class as docs and labels 
        self.training_doc_files = self._get_filenames(params.doc_path)
        self.training_docs, self.training_labels = self.load_data(self.training_doc_files, update_dict=for_train) # TBD: optimize the update dict flag 
        
        # polish and load dictionary/word_to_index/index_to_word as file
        self.num_words = len(self.dictionary)              
        self._updae_word_to_index()
        self._update_docs_dictionary(self.training_docs, 3) # remove low frequency words and add it under the <unknown> key
        
        # save dictionary/word_to_index/index_to_word as file
        np.save(self.dict_path + '_dictionary.npy', self.dictionary)
        np.save(self.dict_path + '_word_to_index.npy', self.word_to_index)
        np.save(self.dict_path + '_index_to_word.npy', self.index_to_word)
        np.save(self.dict_path + '_classes.npy', self.classes)
        # sorted(self.dictionary.items(), key=lambda x:x[1], reverse=True)
        
        # split training / validation docs and show statistics
        num_training = int(len(self.training_docs)*self.data_split)
        data_to_be_fetched = [i for i in range(len(self.training_docs))]
        selected_training_index = data_to_be_fetched[:num_training] #random.sample(data_to_be_fetched, num_training)
        selected_validation_index = list(set(data_to_be_fetched).difference(set(selected_training_index)))
        self.validation_docs = [self.training_docs[x] for x in selected_validation_index]
        self.training_docs = [self.training_docs[x] for x in selected_training_index]
        self.validation_labels = self.training_labels
        print('\nDATASET: %d vocabularies, %d target classes'%(len(self.dictionary), len(self.classes)))
        print('DATASET: %d for training, %d for validation'%(len(self.training_docs), len(self.validation_docs)))
        
        self._update_training_rows_cols() # adapt grid table size according to training docs
         
        # TBD: adjust bbox in @training_docs to eliminate overlaps
        #self.training_docs = self.eliminate_overlap(self.training_docs)
                
        ## 3> call self.next_batch() outside to generate a batch of grid tables data and labels
        self.training_data_tobe_fetched = [i for i in range(len(self.training_docs))]
        self.validation_data_tobe_fetched = [i for i in range(len(self.validation_docs))]        
    
    def _updae_word_to_index(self):
        if self.load_dictionary:
            max_index = len(self.word_to_index.keys())
            for word in self.dictionary:
                if word not in self.word_to_index:
                    max_index += 1
                    self.word_to_index[word] = max_index
                    self.index_to_word[max_index] = word            
        else:   
            self.word_to_index = dict(list(zip(self.dictionary.keys(), list(range(self.num_words))))) 
            self.index_to_word = dict(list(zip(list(range(self.num_words)), self.dictionary.keys())))
    
    def _update_docs_dictionary(self, docs, lower_limit):
        # assign docs words that appear less than @lower_limit times to word <unknown>
        if self.remove_lowfreq_words: 
            for doc in docs:
                for line in doc:
                    [file_name, dressed_text, word_id, [x_left, y_top, x_right, y_bottom], \
                        [image_w, image_h], max_row_words, max_col_words] = line 
                    if self.dictionary[dressed_text] < lower_limit:
                        line = [file_name, '<unknown>', self.word_to_index['<unknown>'], [x_left, y_top, x_right, y_bottom], \
                                [image_w, image_h], max_row_words, max_col_words]
                        self.dictionary[dressed_text] -= 1
                        self.dictionary['<unknown>'] += 1
    
    def next_batch(self):
        #grid_table = np.ones([self.batch_size, self.rows, self.cols, 1])
        #gt_classes = np.ones([self.batch_size, self.rows, self.cols])
        #gt_classes[:,:,0:32] = 0
        # TBD: data augmentation
        
        if len(self.training_data_tobe_fetched) < self.batch_size:
            self.training_data_tobe_fetched = [i for i in range(len(self.training_docs))]            
        selected_index = random.sample(self.training_data_tobe_fetched, self.batch_size)
        self.training_data_tobe_fetched = list(set(self.training_data_tobe_fetched).difference(set(selected_index)))

        training_docs = [self.training_docs[x] for x in selected_index]
        grid_table, gt_classes = self._positional_mapping(training_docs, self.training_labels)
        
        batch = {'grid_table': grid_table, 'gt_classes': gt_classes}
        return batch
    
    def fetch_validation_data(self):
        grid_table, gt_classes = self._positional_mapping(self.validation_docs, self.validation_labels)        
        batch = {'grid_table': grid_table, 'gt_classes': gt_classes}
        return batch
    
    def fetch_test_data(self):        
        grid_table, gt_classes = self._positional_mapping(self.validation_docs, self.validation_labels)        
        batch = {'grid_table': grid_table, 'gt_classes': gt_classes, 'docs': self.validation_docs}
        return batch
    
    def _positional_mapping(self, docs, labels):
        """
        docs in format:
        [[file_name, text, word_id, [x_left, y_top, x_right, y_bottom], [image_w, image_h]], max_row_words, max_col_words ]
        """
        grid_tables = []
        gird_labels = []
        for doc in docs:
            grid_table = np.zeros([self.rows, self.cols], dtype=np.int32)
            grid_label = np.zeros([self.rows, self.cols], dtype=np.int8)
            drawing_board = np.zeros([self.rows, self.cols], dtype=str)
            for item in doc:
                file_name = item[0]
                text = item[1]
                word_id = item[2]
                x_left, y_top, x_right, y_bottom = item[3][:]
                image_w, image_h = item[4][:]
                
                dict_id = self.word_to_index[text]
                    
                class_id = self._dress_class(file_name, word_id, labels)    
                col = int(self.cols * (x_left + (x_right-x_left)/2) / image_w) 
                row = int(self.rows * (y_top + (y_bottom-y_top)/2) / image_h)  
                if grid_label[row, col] == 0: # TBD: overlap avoidance should be done before this
                    grid_table[row, col] = dict_id
                if grid_label[row, col] == 0:
                    grid_label[row, col] = class_id
                    if DEBUG:
                        filler = text if class_id == 0 else str(class_id)+text+'>>' 
                        drawing_board[row, col] = filler
            if DEBUG:
                self.grid_visualization(drawing_board)
            grid_tables.append(np.expand_dims(grid_table, -1)) 
            gird_labels.append(grid_label) 
            
        return grid_tables, gird_labels 
            
    def grid_visualization(self, data):
        import pandas as pd
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')
        fig, ax = plt.subplots()
        ax.xaxis.set_visible(False) 
        ax.yaxis.set_visible(False)
        colLabels = [i for i in range(data.shape[1])]
        ax.table(cellText=data,colLabels=colLabels,loc='center',colLoc='left')#,fontsize=80)
        plt.show()
    
    def load_data(self, data_files, update_dict=False):
        """
        label_dressed in format:
        {file_id: {class: {'key_id':[], 'value_id':[], 'key_text':'', 'value_text':''} } }
        load doc words with location and class returned in format: 
        [[file_name, text, word_id, [x_left, y_top, x_right, y_bottom], [image_w, image_h], max_row_words, max_col_words] ]
        """
        label_dressed = {}
        doc_dressed = []
        if not data_files:
            raise Exception("no data file found.")        
        for file in data_files:
            with open(file, encoding='utf-8') as f:
                data = json.load(f)
                file_id = data['global_attributes']['file_id']
                
                label = self._collect_label(file_id, data['fields'])
                # ignore corrupted data
                if not label:
                    continue
                
                label_dressed.update(label) 
                doc_dressed.append(self._collect_data(file_id, data['text_boxes'], update_dict))
        return doc_dressed, label_dressed       
        
    def _update_training_rows_cols(self):
        self.rows, self.cols = self._cal_rows_cols(self.training_docs)  
        print('DATASHAPE: updated to ({},{}) in DataLoader._update_training_rows_cols() \n'.format(self.rows, self.cols))      
        
    def _cal_rows_cols(self, docs):
        max_row = self.rows
        max_col = self.cols
        for doc in docs:
            for line in doc: 
                _, _, _, _, _, max_row_words, max_col_words = line
                if max_row_words > max_row:
                    max_row = max_row_words
                if max_col_words > max_col:
                    max_col = max_col_words
        rows = (max_row//self.encoding_factor+1) * self.encoding_factor
        cols = (max_col//self.encoding_factor+1) * self.encoding_factor  
        return rows, cols   
    
    def _collect_data(self, file_name, content, update_dict):
        """
        dress and preserve only interested data.
        """
        content_dressed = []
        image_w, image_h, buffer = 0, 0, 2
        for line in content:
            bbox = line['bbox'] # handle data corrupt
            if len(bbox) == 0:
                continue
            
            x_left, y_top, x_right, y_bottom = self._dress_bbox(bbox)        
            # TBD: the real image size is better for calculating the relative x/y/w/h
            if x_right > image_w:
                image_w = x_right + buffer
            if y_bottom > image_h:
                image_h = y_bottom + buffer
                
            word_id = line['id']
            dressed_text, parts = self._dress_text(line['text'], update_dict)
            
            # TBD: seperate bbox according to @dressed_text and @parts
            content_dressed.append([file_name, dressed_text, word_id, [x_left, y_top, x_right, y_bottom]])
            
        num_words_row = [0 for _ in range(image_h)] # number of words in each row
        num_words_col = [0 for _ in range(image_w)] # number of words in each column
        for line in content_dressed:
            _, _, _, [x_left, y_top, x_right, y_bottom] = line
            for y in range(y_top, y_bottom):
                num_words_row[y] += 1
            for x in range(x_left, x_right):
                num_words_col[x] += 1
        max_row_words = max(num_words_row)
        max_col_words = max(num_words_col)
        #print(max_row_words, max_col_words)
            
        for i, line in enumerate(content_dressed): # append height/width/numofwords to the list
            file_name, dressed_text, word_id, [x_left, y_top, x_right, y_bottom] = line
            content_dressed[i] = [file_name, dressed_text, word_id, [x_left, y_top, x_right, y_bottom], \
                                  [image_w, image_h], max_row_words, max_col_words]
        return content_dressed  
    
    def _collect_label(self, file_id, content):
        """
        dress and preserve only interested data.
        label_dressed in format:
        {file_id: {class: {'key_id':[], 'value_id':[], 'key_text':'', 'value_text':''} } }
        """
        label_dressed = dict()
        label_dressed[file_id] = {cls:{} for cls in self.classes[1:]}
        for line in content:
            cls = line['field_name']
            if cls in self.classes:     
                label_dressed[file_id][cls] = \
                    {'key_id':[], 'value_id':[], 'key_text':'', 'value_text':''} 
                label_dressed[file_id][cls]['key_id'] = line['key_id']
                label_dressed[file_id][cls]['value_id'] = line['value_id']
                label_dressed[file_id][cls]['key_text'] = line['key_text'] 
                label_dressed[file_id][cls]['value_text'] = line['value_text']
                
        # handle corrupted data
        for cls in label_dressed[file_id]: 
            if len(label_dressed[file_id][cls]) == 0: # no relvant class in sample @file_id
                continue
            if (len(label_dressed[file_id][cls]['key_text'])>0 and len(label_dressed[file_id][cls]['key_id'])==0) or \
               (len(label_dressed[file_id][cls]['value_text'])>0 and len(label_dressed[file_id][cls]['value_id'])==0):
                return None
            
        return label_dressed

    def _dress_class(self, file_name, word_id, labels):
        """
        label_dressed in format:
        {file_name: {class: {'key_id':[], 'value_id':[], 'key_text':'', 'value_text':''} } }
        """
        if file_name in labels:
            for cls, cls_labels in labels[file_name].items():
                for key, values in cls_labels.items():
                    if (key=='key_id' or key=='value_id') and word_id in values:
                        if key == 'key_id':
                            if self.data_mode == 0:
                                return self.classes.index(cls) * 2 - 1 # odd
                            elif self.data_mode == 1:
                                return self.classes.index(cls)
                            else: # ignore key_id when self.data_mode is not 0 or 1
                                return 0 
                        elif key == 'value_id':
                            if self.data_mode == 0:
                                return self.classes.index(cls) * 2 # even 
                            else: # when self.data_mode is 1 or 2
                                return self.classes.index(cls) 
            return 0 # 0 is of class type 'DontCare'
        print("No matched labels found for {}".format(file_name))
    
    def _dress_text(self, text, update_dict):
        """
        three cases covered: 
        alphabetic string, numeric string, special character
        """
        string = text.lower()
        for i, c in enumerate(string):
            if is_number(c):
                string = string[:i] + '0' + string[i+1:]
                
        if string.isalpha():
            if string in self.special_dict:
                string = self.special_dict[string]
            # TBD: convert a word to its most similar word in a known vocabulary
        elif is_number(string):
            pass
        elif len(string)==1: # special character
            pass
        else:
            # TBD: seperate string as parts for alpha and number combinated strings
            #string = re.findall('[a-z]+', string)
            pass            
        
        if string not in self.dictionary.keys():
            if update_dict:
                self.dictionary[string] = 0
            else:
                string = '<unknown>' # TBD: take special care to unmet words
        if update_dict:
            self.dictionary[string] += 1
        return string, len(string) # len(string) not used   
            
    def _dress_bbox(self, bbox):
        positions = np.array(bbox).reshape([-1])
        x_left = max(0, min(positions[0::2]))
        x_right = max(positions[0::2])
        y_top = max(0, min(positions[1::2]))
        y_bottom = max(positions[1::2])
        w = x_right - x_left
        h = y_bottom - y_top
        return x_left, y_top, x_right, y_bottom   
    
    def _get_filenames(self, data_path):
        files = []
        for dirpath,dirnames,filenames in walk(data_path):
            for filename in filenames:
                file = join(dirpath,filename)
                if file.endswith('csv') or file.endswith('json'):
                    files.append(file)
        return files   
    