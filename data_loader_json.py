# -*- coding: utf-8 -*-
# 2019-01 
# written by Xiaohui Zhao
# xiaohui.zhao@accenture.com
from os import walk
from os.path import isfile, join
import csv, re, random, json
from collections import defaultdict

import numpy as np
import tensorflow as tf
import tokenization

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
    grid tables producer
    """
    def __init__(self, params, update_dict=True, load_dictionary=False, data_split=0.75):
        self.encoding_factor = 8 # ensures the size (rows/cols) of grid table compat with the network
        self.rows_limit = params.target_rows if hasattr(params, 'target_rows') else 64 # handle OOM, must be multiple of self.encoding_factor
        self.cols_limit = params.target_cols if hasattr(params, 'target_cols') else 64 # handle OOM, must be multiple of self.encoding_factor
        self.fill_bbox = params.fill_bbox if hasattr(params, 'fill_bbox') else False # fill bbox with labels or use one single lable for the entire bbox
        self.text_case = params.text_case 
        self.tokenize = params.tokenize
        if self.tokenize:
            self.tokenizer = tokenization.FullTokenizer('dict/vocab.txt', do_lower_case=not self.text_case)
        
        ## 0> parameters to be tuned
        self.load_dictionary = load_dictionary # load dictionary from file rather than start from empty 
        self.dict_path = params.load_dict_from_path if load_dictionary else params.dict_path
        if self.load_dictionary:
            self.dictionary = np.load(self.dict_path + '_dictionary.npy').item()
            self.word_to_index = np.load(self.dict_path + '_word_to_index.npy').item()
            self.index_to_word = np.load(self.dict_path + '_index_to_word.npy').item()
        else:
            self.dictionary = {'[PAD]':0, '[UNK]':0} # word/counts. to be updated in self.load_data() and self._update_docs_dictionary()
            self.word_to_index = {}
            self.index_to_word = {}
        self.classes = ['DontCare', 'VendorName', 'VendorTaxID', 'InvoiceDate', 'InvoiceNumber', 'ExpenseAmount', 'BaseAmount', 'TaxAmount', 'TaxRate']

        self.data_split = data_split # split data to training/validation, 0 for all for validation
        self.data_mode = 2 # 0 to consider key and value as two different class, 1 the same class, 2 only value considered
        self.remove_lowfreq_words = False # remove low frequency words when set as True
        
        self.data_augmentation = params.data_augmentation if hasattr(params, 'data_augmentation') else False # cal rows/cols for each batch of data
        self.data_augmentation_extra = params.data_augmentation_extra if hasattr(params, 'data_augmentation_extra') else False # randomly expand rows/cols
        self.da_extra_rows = params.data_augmentation_extra_rows if hasattr(params, 'data_augmentation_extra_rows') else 0 # randomly expand rows/cols
        self.da_extra_cols = params.data_augmentation_extra_cols if hasattr(params, 'data_augmentation_extra_cols') else 0 # randomly expand rows/cols
        self.rows = 0#32 # to be updated in self._update_training_rows_cols()
        self.cols = 0#32 # to be updated in self._update_training_rows_cols()
        
        self.num_classes = len(self.classes) 
        self.batch_size = params.batch_size if hasattr(params, 'batch_size') else 1        
        
        # TBD: build a special cared dictionary
        self.special_dict = {'0': '[unused10]', '1': '[unused1]', '2': '[unused2]', '3': '[unused3]', '4': '[unused4]', '5': '[unused5]', 
                             '6': '[unused6]', '7': '[unused7]', '8': '[unused8]', '9': '[unused9]'} # map texts to specific tokens        
        
        ## 1.1> load words and their location/class as training/validation docs and labels 
        self.training_doc_files = self._get_filenames(params.doc_path)
        self.training_docs, self.training_labels = self.load_data(self.training_doc_files, update_dict=update_dict) # TBD: optimize the update dict flag
        
        # polish and load dictionary/word_to_index/index_to_word as file
        self.num_words = len(self.dictionary)              
        self._updae_word_to_index()
        self._update_docs_dictionary(self.training_docs, 3, self.remove_lowfreq_words) # remove low frequency words and add it under the <unknown> key
        
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
        print('\n\nDATASET: %d vocabularies, %d target classes'%(len(self.dictionary), len(self.classes)))
        print('DATASET: %d for training, %d for validation'%(len(self.training_docs), len(self.validation_docs)))
        
        ## 1.2> load test files
        self.test_doc_files = self._get_filenames(params.test_path) if hasattr(params, 'test_path') else []
        self.test_docs, self.test_labels = self.load_data(self.test_doc_files, update_dict=update_dict) # TBD: optimize the update dict flag
        print('DATASET: %d for test from %s \n'%(len(self.test_docs), params.test_path if hasattr(params, 'test_path') else '_'))
        
        self.data_shape_statistic() # show data shape static
        if len(self.training_docs) > 0:# adapt grid table size to all training dataset docs 
            self._update_training_rows_cols() 
        else:
            self.rows, self.cols = self.rows_limit, self.cols_limit
        
        # TBD: adjust bbox in @training_docs to eliminate overlaps
        #self.training_docs = self.eliminate_overlap(self.training_docs)
                
        ## 2> call self.next_batch() outside to generate a batch of grid tables data and labels
        self.training_data_tobe_fetched = [i for i in range(len(self.training_docs))]
        self.validation_data_tobe_fetched = [i for i in range(len(self.validation_docs))]        
        self.test_data_tobe_fetched = [i for i in range(len(self.test_docs))]
        
    
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
    
    def _update_docs_dictionary(self, docs, lower_limit, remove_lowfreq_words):
        # assign docs words that appear less than @lower_limit times to word [UNK]
        if remove_lowfreq_words: 
            for doc in docs:
                for line in doc:
                    [file_name, dressed_text, word_id, [x_left, y_top, x_right, y_bottom], \
                        [image_w, image_h], max_row_words, max_col_words] = line 
                    if self.dictionary[dressed_text] < lower_limit:
                        line = [file_name, '[UNK]', self.word_to_index['[UNK]'], [x_left, y_top, x_right, y_bottom], \
                                [image_w, image_h], max_row_words, max_col_words]
                        self.dictionary[dressed_text] -= 1
                        self.dictionary['[UNK]'] += 1
    
    def next_batch(self):
        #grid_table = np.ones([self.batch_size, self.rows, self.cols, 1])
        #gt_classes = np.ones([self.batch_size, self.rows, self.cols])
        #gt_classes[:,:,0:32] = 0
        batch_size = self.batch_size
        
        if len(self.training_data_tobe_fetched) < batch_size:
            self.training_data_tobe_fetched = [i for i in range(len(self.training_docs))]            
        selected_index = random.sample(self.training_data_tobe_fetched, batch_size)
        self.training_data_tobe_fetched = list(set(self.training_data_tobe_fetched).difference(set(selected_index)))

        training_docs = [self.training_docs[x] for x in selected_index]
        
        # data augmentation in each batch
        rows = self.rows
        cols = self.cols
        if self.data_augmentation:
            rows, cols, pre_rows, pre_cols = self._cal_rows_cols(training_docs, extra_augmentation=self.data_augmentation_extra)            
            print('Training grid table augment size: ({},{}) from ({},{})'\
                  .format(rows, cols, pre_rows, pre_cols))
            
        grid_table, gt_classes, bboxes, label_mapids, bbox_mapids, file_name = \
            self._positional_mapping(training_docs, self.training_labels, rows, cols)    
        batch = {'grid_table': grid_table, 'gt_classes': gt_classes, 'bboxes': bboxes, 
                 'label_mapids': label_mapids, 'bbox_mapids': bbox_mapids,
                 'file_name': file_name, 'shape': [rows,cols]}
        return batch
    
    def fetch_validation_data(self):
        batch_size = 1
        
        if len(self.validation_data_tobe_fetched) == 0:
            self.validation_data_tobe_fetched = [i for i in range(len(self.validation_docs))]            
        selected_index = random.sample(self.validation_data_tobe_fetched, 1)
        self.validation_data_tobe_fetched = list(set(self.validation_data_tobe_fetched).difference(set(selected_index)))

        validation_docs = [self.validation_docs[x] for x in selected_index]
        
        rows = self.rows
        cols = self.cols
        ## fixed validation shape leads to better result
        #if self.data_augmentation: # calculate rows/cols for current grid table
        #    rows, cols, _, _ = self._cal_rows_cols(validation_docs, extra_augmentation=False)            
        #    print('Validation grid table real size: ({},{})'.format(rows, cols))
        
        grid_table, gt_classes, bboxes, label_mapids, bbox_mapids, file_name = \
            self._positional_mapping(validation_docs, self.validation_labels, rows, cols)       
        batch = {'grid_table': grid_table, 'gt_classes': gt_classes, 'bboxes': bboxes, 
                 'label_mapids': label_mapids, 'bbox_mapids': bbox_mapids,
                 'file_name': file_name, 'shape': [rows,cols]}
        return batch
    
    def fetch_test_data(self): 
        batch_size = 1
        
        if len(self.test_data_tobe_fetched) == 0:
            self.test_data_tobe_fetched = [i for i in range(len(self.test_docs))]
            return None
                    
        selected_index = self.test_data_tobe_fetched[0]
        self.test_data_tobe_fetched = list(set(self.test_data_tobe_fetched).difference(set([selected_index])))

        test_docs = [self.test_docs[selected_index]]
        
        rows = self.rows
        cols = self.cols
        #if self.data_augmentation:
        #    rows, cols, _, _ = self._cal_rows_cols(test_docs, extra_augmentation=False)            
        #    print('Test grid table real size: ({},{})'.format(rows, cols))
        #if len(self.test_docs) % 100: # show static every 100        
        #    print('Test grid table size: ({},{}), {} left to be tested'.format(rows, cols, len(self.test_data_tobe_fetched)))
            
        grid_table, gt_classes, bboxes, label_mapids, bbox_mapids, file_name = \
            self._positional_mapping(test_docs, self.test_labels, rows, cols)        
        batch = {'grid_table': grid_table, 'gt_classes': gt_classes, 'bboxes': bboxes, 
                 'label_mapids': label_mapids, 'bbox_mapids': bbox_mapids,
                 'file_name': file_name, 'shape': [rows,cols]}
        return batch
    
    def data_shape_statistic(self):        
        def shape_statistic(docs):
            res_all = defaultdict(int)
            res_row = defaultdict(int)
            res_col = defaultdict(int)
            for doc in docs:
                rows, cols, _, _ = self._cal_rows_cols([doc])
                res_all[rows] += 1
                res_all[cols] += 1
                res_row[rows] += 1
                res_col[cols] += 1
            sorted(res_all.items(), key=lambda x:x[1], reverse=True)
            sorted(res_row.items(), key=lambda x:x[1], reverse=True)
            sorted(res_col.items(), key=lambda x:x[1], reverse=True)
            return res_all, res_row, res_col
    
        tss, tss_r, tss_c = shape_statistic(self.training_docs) # training shape static
        vss, vss_r, vss_c = shape_statistic(self.validation_docs)
        tess, tess_r, tess_c = shape_statistic(self.test_docs)
        print("Training shape statistic: ", tss)
        print("\t rows statistic: ", tss_r)
        print("\t cols statistic: ", tss_c)
        print("Validation shape statistic: ", vss)
        print("\t rows statistic: ", vss_r)
        print("\t cols statistic: ", vss_c)
        print("Test shape statistic: ", tess)
        print("\t rows statistic: ", tess_r)
        print("\t cols statistic: ", tess_c)
        
        ## remove data samples not matching the training principle
        def data_laundry(docs):
            idx = 0
            while idx < len(docs):
                rows, cols, _, _ = self._cal_rows_cols([docs[idx]])
                if rows > self.rows_limit or cols > self.cols_limit:
                    del docs[idx]
                else:
                    idx += 1
        data_laundry(self.training_docs)
        data_laundry(self.validation_docs)
        data_laundry(self.training_docs)
        print("Grids larger than ({},{}) removed".format(self.rows_limit, self.cols_limit))
        
        tss, tss_r, tss_c = shape_statistic(self.training_docs) # training shape static
        vss, vss_r, vss_c = shape_statistic(self.validation_docs)
        tess, tess_r, tess_c = shape_statistic(self.test_docs)
        print("Training shape statistic after laundary: ", tss)
        print("\t rows statistic: ", tss_r)
        print("\t cols statistic: ", tss_c)
        print("Validation shape statistic after laundary: ", vss)
        print("\t rows statistic: ", vss_r)
        print("\t cols statistic: ", vss_c)
        print("Test shape statistic after laundary: ", tess)
        print("\t rows statistic: ", tess_r)
        print("\t cols statistic: ", tess_c)
    
    def _positional_mapping(self, docs, labels, rows, cols):
        """
        docs in format:
        [[file_name, text, word_id, [x_left, y_top, x_right, y_bottom], [image_w, image_h]], max_row_words, max_col_words ]
        return grid_tables, gird_labels, dict bboxes {file_name:[]}, file_names
        """
        grid_tables = []
        gird_labels = []
        bboxes = {}
        label_mapids = []
        bbox_mapids = [] # [{}, ] bbox identifier, each id with one or multiple bbox/bboxes
        file_names = []
        for doc in docs:
            grid_table = np.zeros([rows, cols], dtype=np.int32)
            grid_label = np.zeros([rows, cols], dtype=np.int8)
            bbox = [[[] for c in range(cols)] for r in range(rows)]
            bbox_id, bbox_mapid = 0, {} # one word in one or many positions in a bbox is mapped in bbox_mapid
            label_mapid = [[] for _ in range(self.num_classes)] # each class is connected to several bboxes (words)
            drawing_board = np.zeros([rows, cols], dtype=str)
            for item in doc:
                file_name = item[0]
                text = item[1]
                word_id = item[2]
                x_left, y_top, x_right, y_bottom = item[3][:]
                image_w, image_h = item[4][:]
                
                dict_id = self.word_to_index[text]                
                class_id = self._dress_class(file_name, word_id, labels)
                
                bbox_id += 1
                if self.fill_bbox: # TBD: overlap avoidance
                    top = int(rows * y_top / image_h)
                    bottom = int(rows * y_bottom / image_h)
                    left = int(cols * x_left / image_w)
                    right = int(cols * x_right / image_w)
                    grid_table[top:bottom, left:right] = dict_id  
                    grid_label[top:bottom, left:right] = class_id  
                    
                    label_mapid[class_id].append(bbox_id)
                    for row in range(top, bottom):
                        for col in range(left, right):
                            bbox_mapid[row*cols+col] = bbox_id
                    
                    for y in range(top, bottom):
                        for x in range(left, right):
                            bbox[y][x] = [x_left, y_top, x_right-x_left, y_bottom-y_top]
                else:
                    col = int(cols * (x_left + (x_right-x_left)/2) / image_w) 
                    row = int(rows * (y_top + (y_bottom-y_top)/2) / image_h)  
                    if grid_label[row, col] == 0:
                        grid_table[row, col] = dict_id
                        grid_label[row, col] = class_id
                        
                        label_mapid[class_id].append(bbox_id)
                        bbox_mapid[row*cols+col] = bbox_id
                        
                        bbox[row][col] = [x_left, y_top, x_right-x_left, y_bottom-y_top]
                        
                if DEBUG:
                    filler = text if class_id == 0 else str(class_id)+text+'>>' 
                    drawing_board[row, col] = filler
            if DEBUG:
                self.grid_visualization(drawing_board)
            grid_tables.append(np.expand_dims(grid_table, -1)) 
            gird_labels.append(grid_label) 
            bboxes[file_name] = bbox
            label_mapids.append(label_mapid)
            bbox_mapids.append(bbox_mapid)
            file_names.append(file_name)
            
        return grid_tables, gird_labels, bboxes, label_mapids, bbox_mapids, file_names
    
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
            print("no data file found.")        
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
        self.rows, self.cols, _, _ = self._cal_rows_cols(self.training_docs)  
        print('\nDATASHAPE: data set with maximum grid table of ({},{}), updated in DataLoader._update_training_rows_cols()\n'.format(self.rows, self.cols))      
        
    def _cal_rows_cols(self, docs, extra_augmentation=False):
        max_row = 0
        max_col = 0
        for doc in docs:
            for line in doc: 
                _, _, _, _, _, max_row_words, max_col_words = line
                if max_row_words > max_row:
                    max_row = max_row_words
                if max_col_words > max_col:
                    max_col = max_col_words                    
        pre_rows = (max_row//self.encoding_factor+1) * self.encoding_factor
        pre_cols = (max_col//self.encoding_factor+1) * self.encoding_factor
        
        pad_row, pad_col = 0, 0
        if extra_augmentation:
            pad_row = abs(int(random.gauss(0, self.da_extra_rows*self.encoding_factor))) #abs(random.gauss(0, u))
            pad_col = abs(int(random.gauss(0, self.da_extra_cols*self.encoding_factor))) #random.randint(0, u)
            rows = min(((max_row+pad_row)//self.encoding_factor+1) * self.encoding_factor, self.rows_limit) # apply upper boundary to avoid OOM
            cols = min(((max_col+pad_col)//self.encoding_factor+1) * self.encoding_factor, self.cols_limit) # apply upper boundary to avoid OOM
        else:
            rows = pre_rows
            cols = pre_cols
        return rows, cols, pre_rows, pre_cols # 5x upper boundary to avoid OOM
    
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
            dressed_texts = self._dress_text(line['text'], update_dict)
            
            # TBD: seperate bbox according to @dressed_text and @parts
            num_block = len(dressed_texts)
            for i, dressed_text in enumerate(dressed_texts): # for loop is used for handling tokenized text
                x_left = int(x_left + (x_right-x_left) / num_block * (i))
                x_right = int(x_left + (x_right-x_left) / num_block * (i+1))
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
        string = text if self.text_case else text.lower()
        for i, c in enumerate(string):
            if is_number(c):
                string = string[:i] + '0' + string[i+1:]
                
        strings = [string]
        if self.tokenize:
            strings = self.tokenizer.tokenize(strings[0])
            #print(string, '-->', strings)
            
        for idx, string in enumerate(strings):            
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
                    print('unknown text: ' + string)
                    string = '[UNK]' # TBD: take special care to unmet words\
            self.dictionary[string] += 1
            
            strings[idx] = string
        return strings
            
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