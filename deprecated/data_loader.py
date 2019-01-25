# -*- coding: utf-8 -*-
# 2019-01 
# written by Xiaohui Zhao
# xiaohui.zhao@accenture.com
from os import walk
from os.path import isfile, join
import csv, re, random

import numpy as np
import tensorflow as tf

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
    def __init__(self, params):
        ## 0> parameters to be tuned
        self.dictionary = {'<dontcare>':0, '<unknown>':1, '<name>':2} # to be updated in self.load_data()
        self.classes = ['DontCare', 'VendorName', 'VendorTaxID', 'InvoiceDate', 'InvoiceNumber', 'ExpenseAmount']
        self.combinekv = True # False to consider key and value as two different class, otherwise the same class
        self.num_classes = len(self.classes) 
        
        self.rows = 32 # to be updated in self._update_training_rows_cols()
        self.cols = 32 # to be updated in self._update_training_rows_cols()
        self.encoding_factor = 8 # ensures the size (rows/cols) of grid table compat with the network
        
        self.batch_size = params.batch_size        
        
        # TBD: build a special cared dictionary
        self.special_dict = {'some_word': '<dontcare>'} # map texts to specific tokens        
        
        ## 1> load words and their location/class as docs and labels 
        self.training_doc_files = self._get_filenames(params.training_doc_path)
        self.training_label_files = self._get_filenames(params.training_label_path)
        self.training_docs, self.training_labels = self.load_data(self.training_doc_files, self.training_label_files, True)  
        
        self.validation_doc_files = self._get_filenames(params.validation_doc_path)
        self.validation_label_files = self._get_filenames(params.validation_label_path)
        self.validation_docs, self.validation_labels = self.load_data(self.validation_doc_files, self.validation_label_files)  
        
        self.num_words = len(self.dictionary) # TBD: check correctness of dictionary
        sorted(self.dictionary.items(), key=lambda x:x[1], reverse=True)
        self.word_to_index = dict(list(zip(self.dictionary.keys(), list(range(self.num_words))))) 
        self.index_to_word = dict(list(zip(list(range(self.num_words)), self.dictionary.keys())))  
        
        self._update_training_rows_cols() # adapt grid table size according to training data
         
        # TBD: adjust bbox in @training_docs to eliminate overlaps
        #self.training_docs = self.eliminate_overlap(self.training_docs)
        
        ## 2> call self.next_batch() outside to generate a batch of grid tables data and labels
        self.training_data_tobe_fetched = [i for i in range(len(self.training_docs))]
    
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
        doc_files = self._get_filenames(params.test_doc_path)
        docs, _ = self.load_data(doc_files, None)
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
    
    def load_data(self, data_files, label_files, update_dict=False):
        """
        load words with location and class returned in format: 
        [[file_name, text, word_id, [x_left, y_top, x_right, y_bottom], [image_w, image_h], max_row_words, max_col_words] ]
        """
        label_dressed = {}
        if label_files:
            if not label_files:
                raise Exception("no label file found.")
        
            for file in label_files:
                with open(file, encoding='utf-8') as f:
                #with open(file) as f: # .decode("ISO-8859-1")
                    content_csv = csv.DictReader(f)
                    label_dressed.update(self._collect_label(file, content_csv))     
        else:
            print('no label file provided')
        
        doc_dressed = []
        if not data_files:
            raise Exception("no data file found.")        
        for file in data_files:
            with open(file, encoding='utf-8') as f:
            #with open(file) as f:
                content_csv = csv.DictReader(f)
                doc_dressed.append(self._collect_data(file, content_csv, update_dict))

        return doc_dressed, label_dressed        
        
    def _update_training_rows_cols(self):
        self.rows, self.cols = self._cal_rows_cols(self.training_docs)  
        print('(rows,cols) is updated to ({},{}) in DataLoader._update_training_rows_cols() \n'.format(self.rows, self.cols))      
        
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
            x_left, y_top, x_right, y_bottom = self._dress_bbox(line['bbox'])        
            # TBD: the real image size is better for calculating the relative x/y/w/h
            if x_right > image_w:
                image_w = x_right + buffer
            if y_bottom > image_h:
                image_h = y_bottom + buffer
                
            word_id = line['word_id']
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
    
    def _collect_label(self, file_name, content):
        """
        dress and preserve only interested data.
        label_dressed in format:
        {task_id: {class: {'key_id':[], 'value_id':[], 'key_text':'', 'value_text':''} } }
        """
        label_dressed = dict()
        for line in content:
            task_id = line['task_id']
            if task_id not in label_dressed:
                label_dressed[task_id] = {cls:{} for cls in self.classes[1:]}
            
            cls = line['field_name']
            if cls in self.classes:     
                label_dressed[task_id][cls] = \
                    {'key_id':[], 'value_id':[], 'key_text':'', 'value_text':''} 
                label_dressed[task_id][cls]['key_id'] = re.findall('\d+', line['key_id_x'])
                label_dressed[task_id][cls]['value_id'] = re.findall('\d+', line['value_id_x'])
                label_dressed[task_id][cls]['key_text'] = line['key_text_x'] # not used
                label_dressed[task_id][cls]['value_text'] = line['value_text_x'] # not used
        return label_dressed

    def _dress_class(self, file_name, word_id, labels):
        """
        label_dressed in format:
        {task_id: {class: {'key_id':[], 'value_id':[], 'key_text':'', 'value_text':''} } }
        """
        task_id = re.findall('\d+', file_name)[-1]
        if task_id in labels:
            for cls, cls_labels in labels[task_id].items():
                for key, values in cls_labels.items():
                    if word_id in values:
                        if key == 'key_id':
                            return self.classes.index(cls) if self.combinekv else self.classes.index(cls) * 2 - 1 # odd 
                        elif key == 'value_id':
                            return self.classes.index(cls) if self.combinekv else self.classes.index(cls) * 2 # even 
            return 0 # 0 is of class type 'DontCare'
        print("No matched labels found for {}".format(task_id))
    
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
        positions = [int(x) for x in re.findall('\d+', bbox)]
        x_left = min(positions[0::2])
        x_right = max(positions[0::2])
        y_top = min(positions[1::2])
        y_bottom = max(positions[1::2])
        w = x_right - x_left
        h = y_bottom - y_top
#         if x_max//self.quantization_factor > self.cols:
#             self.cols = (x_max//self.encoding_factor+1) * self.encoding_factor
#         if y_max//self.quantization_factor > self.rows:
#             self.rows = (y_max//self.encoding_factor+1) * self.encoding_factor
        return x_left, y_top, x_right, y_bottom   
    
    def _get_filenames(self, data_path):
        files = []
        for dirpath,dirnames,filenames in walk(data_path):
            for filename in filenames:
                file = join(dirpath,filename)
                if file.endswith('csv'):
                    files.append(file)
        return files   
    