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
import cv2

DEBUG = False # True to show grid as image 

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
        self.random = False
        self.data_laundry = False
        self.encoding_factor = 1 # ensures the size (rows/cols) of grid table compat with the network
        #self.classes = ['DontCare', 'Table']
        self.classes = ['DontCare', 'Column']
        #self.classes = ['DontCare', 'VendorName', 'VendorTaxID', 'InvoiceDate', 'InvoiceNumber', 'ExpenseAmount', 'BaseAmount', 'TaxAmount', 'TaxRate']
        
        self.doc_path = params.doc_path
        self.doc_test_path = params.test_path
        self.use_cutie2 = params.use_cutie2 
        self.text_case = params.text_case 
        self.tokenize = params.tokenize
        if self.tokenize:
            self.tokenizer = tokenization.FullTokenizer('dict/vocab.txt', do_lower_case=not self.text_case)
        
        self.rows = self.encoding_factor # to be updated 
        self.cols = self.encoding_factor # to be updated 
        self.segment_grid = params.segment_grid if hasattr(params, 'segment_grid') else False # segment grid into two parts if grid is larger than cols_target
        self.augment_strategy = params.augment_strategy if hasattr(params, 'augment_strategy') else 1 
        self.pm_strategy = params.positional_mapping_strategy if hasattr(params, 'positional_mapping_strategy') else 2 
        self.rows_segment = params.rows_segment if hasattr(params, 'rows_segment') else 72 
        self.cols_segment = params.cols_segment if hasattr(params, 'cols_segment') else 72
        self.rows_target = params.rows_target if hasattr(params, 'rows_target') else 64 
        self.cols_target = params.cols_target if hasattr(params, 'cols_target') else 64 
        self.rows_ulimit = params.rows_ulimit if hasattr(params, 'rows_ulimit') else 80 # handle OOM, must be multiple of self.encoding_factor
        self.cols_ulimit = params.cols_ulimit if hasattr(params, 'cols_ulimit') else 80 # handle OOM, must be multiple of self.encoding_factor
                
        self.fill_bbox = params.fill_bbox if hasattr(params, 'fill_bbox') else False # fill bbox with labels or use one single lable for the entire bbox
        
        self.data_augmentation_dropout = params.data_augmentation_dropout if hasattr(params, 'data_augmentation_dropout') else False # TBD: randomly dropout rows/cols
        self.data_augmentation_extra = params.data_augmentation_extra if hasattr(params, 'data_augmentation_extra') else False # randomly expand rows/cols
        self.da_extra_rows = params.data_augmentation_extra_rows if hasattr(params, 'data_augmentation_extra_rows') else 0 # randomly expand rows/cols
        self.da_extra_cols = params.data_augmentation_extra_cols if hasattr(params, 'data_augmentation_extra_cols') else 0 # randomly expand rows/cols
        
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

        self.data_split = data_split # split data to training/validation, 0 for all for validation
        self.data_mode = 2 # 0 to consider key and value as two different class, 1 the same class, 2 only value considered
        self.remove_lowfreq_words = False # remove low frequency words when set as True
        
        self.num_classes = len(self.classes) 
        self.batch_size = params.batch_size if hasattr(params, 'batch_size') else 1        
        
        # TBD: build a special cared dictionary
        self.special_dict = {'*', '='} # map texts to specific tokens        
        
        ## 1.1> load words and their location/class as training/validation docs and labels 
        self.training_doc_files = self._get_filenames(self.doc_path)
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
        selected_training_index = data_to_be_fetched[:num_training] 
        if self.random:
            selected_training_index = random.sample(data_to_be_fetched, num_training)
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
            self.rows, self.cols, _, _ = self._cal_rows_cols(self.training_docs)  
            print('\nDATASHAPE: data set with maximum grid table of ({},{}), updated.\n'.format(self.rows, self.cols))    
        else:
            self.rows, self.cols = self.rows_ulimit, self.cols_ulimit
                
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
        batch_size = self.batch_size
        
        while True:
            if len(self.training_data_tobe_fetched) < batch_size:
                self.training_data_tobe_fetched = [i for i in range(len(self.training_docs))]            
            selected_index = random.sample(self.training_data_tobe_fetched, batch_size)
            self.training_data_tobe_fetched = list(set(self.training_data_tobe_fetched).difference(set(selected_index)))
    
            training_docs = [self.training_docs[x] for x in selected_index]
            
            ## data augmentation in each batch if self.data_augmentation==True
            rows, cols, pre_rows, pre_cols = self._cal_rows_cols(training_docs, extra_augmentation=self.data_augmentation_extra, dropout=self.data_augmentation_dropout)
            if self.data_augmentation_extra:
                print('Training grid AUGMENT size: ({},{}) from ({},{})'\
                      .format(rows, cols, pre_rows, pre_cols))
            
            grid_table, gt_classes, bboxes, label_mapids, bbox_mapids, file_names, updated_cols, ps_indices_x, ps_indices_y = \
                self._positional_mapping(training_docs, self.training_labels, rows, cols)   
            if updated_cols > cols:
                print('Training grid EXPAND size: ({},{}) from ({},{})'\
                      .format(rows, updated_cols, rows, cols))
                grid_table, gt_classes, bboxes, label_mapids, bbox_mapids, file_names, _, ps_indices_x, ps_indices_y = \
                    self._positional_mapping(training_docs, self.training_labels, rows, updated_cols, update_col=False)  
            
            ## load image and generate corresponding @ps_1dindices
            images, ps_1d_indices = [], []
            if self.use_cutie2:
                images, ps_1d_indices = self._positional_sampling(self.doc_path, file_names, ps_indices_x, ps_indices_y, updated_cols)   
                #print("image fetched {}".format(len(images)))          
                if len(images) == batch_size:
                    break
            else:
                break
        
        batch = {'grid_table': np.array(grid_table), 'gt_classes': np.array(gt_classes), 
                 'data_image': np.array(images), 'ps_1d_indices': np.array(ps_1d_indices), # @images and @ps_1d_indices are only used for CUTIEv2
                 'bboxes': bboxes, 'label_mapids': label_mapids, 'bbox_mapids': bbox_mapids,
                 'file_name': file_names, 'shape': [rows,cols]}
        return batch
    
    def fetch_validation_data(self):
        batch_size = 1
        
        while True:
            if len(self.validation_data_tobe_fetched) == 0:
                self.validation_data_tobe_fetched = [i for i in range(len(self.validation_docs))]            
            selected_index = random.sample(self.validation_data_tobe_fetched, 1)
            self.validation_data_tobe_fetched = list(set(self.validation_data_tobe_fetched).difference(set(selected_index)))
    
            validation_docs = [self.validation_docs[x] for x in selected_index]
            
            ## fixed validation shape leads to better result (to be verified)
            real_rows, real_cols, _, _ = self._cal_rows_cols(validation_docs, extra_augmentation=False)
            rows = max(self.rows_target, real_rows)
            cols = max(self.rows_target, real_cols)
            
            grid_table, gt_classes, bboxes, label_mapids, bbox_mapids, file_names, updated_cols, ps_indices_x, ps_indices_y = \
                self._positional_mapping(validation_docs, self.validation_labels, rows, cols)   
            if updated_cols > cols:
                print('Validation grid EXPAND size: ({},{}) from ({},{})'\
                      .format(rows, updated_cols, rows, cols))
                grid_table, gt_classes, bboxes, label_mapids, bbox_mapids, file_names, _, ps_indices_x, ps_indices_y = \
                    self._positional_mapping(validation_docs, self.validation_labels, rows, updated_cols, update_col=False)     
                    
            ## load image and generate corresponding @ps_1dindices
            images, ps_1d_indices = [], []
            if self.use_cutie2:
                images, ps_1d_indices = self._positional_sampling(self.doc_path, file_names, ps_indices_x, ps_indices_y, updated_cols)  
                if len(images) == batch_size:
                    break        
            else:
                break  
        
        batch = {'grid_table': np.array(grid_table), 'gt_classes': np.array(gt_classes), 
                 'data_image': np.array(images), 'ps_1d_indices': np.array(ps_1d_indices), # @images and @ps_1d_indices are only used for CUTIEv2
                 'bboxes': bboxes, 'label_mapids': label_mapids, 'bbox_mapids': bbox_mapids,
                 'file_name': file_names, 'shape': [rows,cols]}
        return batch
    
    def fetch_test_data(self): 
        batch_size = 1
        
        while True:
            if len(self.test_data_tobe_fetched) == 0:
                self.test_data_tobe_fetched = [i for i in range(len(self.test_docs))]
                return None
                        
            selected_index = self.test_data_tobe_fetched[0]
            self.test_data_tobe_fetched = list(set(self.test_data_tobe_fetched).difference(set([selected_index])))
    
            test_docs = [self.test_docs[selected_index]]
            
            real_rows, real_cols, _, _ = self._cal_rows_cols(test_docs, extra_augmentation=False)
            rows = max(self.rows_target, real_rows) # small shaped documents have better performance with shape 64
            cols = max(self.cols_target, real_cols) # large shaped docuemnts have better performance with shape 80
                
            grid_table, gt_classes, bboxes, label_mapids, bbox_mapids, file_names, updated_cols, ps_indices_x, ps_indices_y = \
                self._positional_mapping(test_docs, self.test_labels, rows, cols)   
            if updated_cols > cols:
                print('Test grid EXPAND size: ({},{}) from ({},{})'\
                      .format(rows, updated_cols, rows, cols))
                grid_table, gt_classes, bboxes, label_mapids, bbox_mapids, file_names, _, ps_indices_x, ps_indices_y = \
                    self._positional_mapping(test_docs, self.test_labels, rows, updated_cols, update_col=False)    
                    
            ## load image and generate corresponding @ps_1dindices
            images, ps_1d_indices = [], []
            if self.use_cutie2:
                images, ps_1d_indices = self._positional_sampling(self.doc_test_path, file_names, ps_indices_x, ps_indices_y, updated_cols)          
                if len(images) == batch_size:
                    break          
            else:
                break
        
        batch = {'grid_table': np.array(grid_table), 'gt_classes': np.array(gt_classes), 
                 'data_image': np.array(images), 'ps_1d_indices': np.array(ps_1d_indices), # @images and @ps_1d_indices are only used for CUTIEv2
                 'bboxes': bboxes, 'label_mapids': label_mapids, 'bbox_mapids': bbox_mapids,
                 'file_name': file_names, 'shape': [rows,cols]}
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
            res_all = sorted(res_all.items(), key=lambda x:x[0], reverse=True)
            res_row = sorted(res_row.items(), key=lambda x:x[0], reverse=True)
            res_col = sorted(res_col.items(), key=lambda x:x[0], reverse=True)
            return res_all, res_row, res_col
    
        tss, tss_r, tss_c = shape_statistic(self.training_docs) # training shape static
        vss, vss_r, vss_c = shape_statistic(self.validation_docs)
        tess, tess_r, tess_c = shape_statistic(self.test_docs)
        print("Training statistic: ", tss)
        print("\t num: ", len(self.training_docs))
        print("\t rows statistic: ", tss_r)
        print("\t cols statistic: ", tss_c)
        print("\nValidation statistic: ", vss)
        print("\t num: ", len(self.validation_docs))
        print("\t rows statistic: ", vss_r)
        print("\t cols statistic: ", vss_c)
        print("\nTest statistic: ", tess)
        print("\t num: ", len(self.test_docs))
        print("\t rows statistic: ", tess_r)
        print("\t cols statistic: ", tess_c)
        
        ## remove data samples not matching the training principle
        def data_laundry(docs):
            idx = 0
            while idx < len(docs):
                rows, cols, _, _ = self._cal_rows_cols([docs[idx]])
                if rows > self.rows_ulimit or cols > self.cols_ulimit:
                    del docs[idx]
                else:
                    idx += 1
        if self.data_laundry:
            print("\nRemoving grids with shape larger than ({},{}).".format(self.rows_ulimit, self.cols_ulimit))
            data_laundry(self.training_docs)
            data_laundry(self.validation_docs)
            data_laundry(self.training_docs)
        
            tss, tss_r, tss_c = shape_statistic(self.training_docs) # training shape static
            vss, vss_r, vss_c = shape_statistic(self.validation_docs)
            tess, tess_r, tess_c = shape_statistic(self.test_docs)
            print("Training statistic after laundary: ", tss)
            print("\t num: ", len(self.training_docs))
            print("\t rows statistic: ", tss_r)
            print("\t cols statistic: ", tss_c)
            print("Validation statistic after laundary: ", vss)
            print("\t num: ", len(self.validation_docs))
            print("\t rows statistic: ", vss_r)
            print("\t cols statistic: ", vss_c)
            print("Test statistic after laundary: ", tess)
            print("\t num: ", len(self.test_docs))
            print("\t rows statistic: ", tess_r)
            print("\t cols statistic: ", tess_c)
    
    def _positional_mapping(self, docs, labels, rows, cols):
        """
        docs in format:
        [[file_name, text, word_id, [x_left, y_top, x_right, y_bottom], [left, top, right, bottom], max_row_words, max_col_words] ]
        return grid_tables, gird_labels, dict bboxes {file_name:[]}, file_names
        """
        grid_tables = []
        gird_labels = []
        ps_indices_x = [] # positional sampling indices
        ps_indices_y = [] # positional sampling indices
        bboxes = {}
        label_mapids = []
        bbox_mapids = [] # [{}, ] bbox identifier, each id with one or multiple bbox/bboxes
        file_names = []
        for doc in docs:
            items = []
            cols_e = 2 * cols # use @cols_e larger than required @cols as buffer
            grid_table = np.zeros([rows, cols_e], dtype=np.int32)
            grid_label = np.zeros([rows, cols_e], dtype=np.int8)
            ps_x = np.zeros([rows, cols_e], dtype=np.int32)
            ps_y = np.zeros([rows, cols_e], dtype=np.int32)
            bbox = [[] for c in range(cols_e) for r in range(rows)]
            bbox_id, bbox_mapid = 0, {} # one word in one or many positions in a bbox is mapped in bbox_mapid
            label_mapid = [[] for _ in range(self.num_classes)] # each class is connected to several bboxes (words)
            drawing_board = np.zeros([rows, cols_e], dtype=str)
            for item in doc:
                file_name = item[0]
                text = item[1]
                word_id = item[2]
                x_left, y_top, x_right, y_bottom = item[3][:]
                left, top, right, bottom = item[4][:]
                
                dict_id = self.word_to_index[text]                
                entity_id, class_id = self._dress_class(file_name, word_id, labels)
                
                bbox_id += 1
#                 if self.fill_bbox: # TBD: overlap avoidance
#                     top = int(rows * y_top / image_h)
#                     bottom = int(rows * y_bottom / image_h)
#                     left = int(cols * x_left / image_w)
#                     right = int(cols * x_right / image_w)
#                     grid_table[top:bottom, left:right] = dict_id  
#                     grid_label[top:bottom, left:right] = class_id  
#                      
#                     label_mapid[class_id].append(bbox_id)
#                     for row in range(top, bottom):
#                         for col in range(left, right):
#                             bbox_mapid[row*cols+col] = bbox_id
#                      
#                     for y in range(top, bottom):
#                         for x in range(left, right):
#                             bbox[y][x] = [x_left, y_top, x_right-x_left, y_bottom-y_top]
                label_mapid[class_id].append(bbox_id)    
                
                #v_c = (y_top - top + (y_bottom-y_top)/2) / (bottom-top)
                #h_c = (x_left - left + (x_right-x_left)/2) / (right-left)
                #v_c = (y_top + (y_bottom-y_top)/2) / bottom
                #h_c = (x_left + (x_right-x_left)/2) / right 
                #v_c = (y_top-top) / (bottom-top)
                #h_c = (x_left-left) / (right-left)
                #v_c = (y_top) / (bottom)
                #h_c = (x_left) / (right)
                box_y = y_top + (y_bottom-y_top)/2
                box_x = x_left # h_l is used for image feature map positional sampling
                v_c = (y_top - top + (y_bottom-y_top)/2) / (bottom-top)
                h_c = (x_left - left + (x_right-x_left)/2) / (right-left) # h_c is used for sorting items
                row = int(rows * v_c) 
                col = int(cols * h_c) 
                items.append([row, col, [box_y, box_x], [v_c, h_c], file_name, dict_id, class_id, entity_id, bbox_id, [x_left, y_top, x_right-x_left, y_bottom-y_top]])                       
            
            items.sort(key=lambda x: (x[0], x[3], x[5])) # sort according to row > h_c > bbox_id
            for item in items:
                row, col, [box_y, box_x], [v_c, h_c], file_name, dict_id, class_id, entity_id, bbox_id, box = item
                entity_class_id = entity_id*self.num_classes + class_id
                
                while col < cols and grid_table[row, col] != 0:
                    col += 1            
                
                # self.pm_strategy 0: skip if overlap
                # self.pm_strategy 1: shift to find slot if overlap
                # self.pm_strategy 2: expand grid table if overlap
                if self.pm_strategy == 0:
                    if col == cols:                     
                        print('overlap in {} row {} r{}c{}!'.
                              format(file_name, row, rows, cols))
                        #print(grid_table[row,:])
                        #print('overlap in {} <{}> row {} r{}c{}!'.
                        #      format(file_name, self.index_to_word[dict_id], row, rows, cols))
                    else:
                        grid_table[row, col] = dict_id
                        grid_label[row, col] = entity_class_id                       
                        bbox_mapid[row*cols+col] = bbox_id                       
                        bbox[row*cols+col] = box   
                elif self.pm_strategy==1 or self.pm_strategy==2:
                    ptr = 0
                    if col == cols: # shift to find slot to drop the current item
                        col -= 1
                        while ptr<cols and grid_table[row, ptr] != 0:
                            ptr += 1
                        if ptr == cols:
                            grid_table[row, :-1] = grid_table[row, 1:]
                        else:
                            grid_table[row, ptr:-1] = grid_table[row, ptr+1:]
                        
                    if self.pm_strategy == 2:
                        while col < cols_e and grid_table[row, col] != 0:
                            col += 1
                        if col > cols: # update maximum cols in current grid
                            print(grid_table[row,:col])
                            print('overlap in {} <{}> row {} r{}c{}!'.
                                  format(file_name, self.index_to_word[dict_id], row, rows, cols))
                            cols = col
                        if col == cols_e:      
                            print('overlap!')
                    
                    grid_table[row, col] = dict_id
                    grid_label[row, col] = entity_class_id
                    ps_x[row, col] = box_x
                    ps_y[row, col] = box_y
                    bbox_mapid[row*cols+col] = bbox_id     
                    bbox[row*cols+col] = box
                
            cols = self._fit_shape(cols)
            grid_table = grid_table[..., :cols]
            grid_label = grid_label[..., :cols]
            ps_x = np.array(ps_x[..., :cols])
            ps_y = np.array(ps_y[..., :cols])
            
            if DEBUG:
                self.grid_visualization(file_name, grid_table, grid_label)
            
            grid_tables.append(np.expand_dims(grid_table, -1)) 
            gird_labels.append(grid_label) 
            ps_indices_x.append(ps_x)
            ps_indices_y.append(ps_y)
            bboxes[file_name] = bbox
            label_mapids.append(label_mapid)
            bbox_mapids.append(bbox_mapid)
            file_names.append(file_name)
            
        return grid_tables, gird_labels, bboxes, label_mapids, bbox_mapids, file_names, cols, ps_indices_x, ps_indices_y
    
    def _positional_sampling(self, path, file_names, ps_indices_x, ps_indices_y, updated_cols):
        images, ps_1d_indices = [], []
        
        ## load image and generate corresponding @ps_1dindices
        max_h, max_w = 0, updated_cols
        for i in range(len(file_names)):
            file_name = file_names[i]
            file_path = join(path, file_name) # TBD: ensure image is upright
            ps_1d_x = np.array(ps_indices_x[i], dtype=np.float32).reshape([-1])
            ps_1d_y = np.array(ps_indices_y[i], dtype=np.float32).reshape([-1])
            
            image = cv2.imread(file_path)
            if image is not None:
                h, w, _ = image.shape # [h,w,c]
                factor = max_w / w
                
                h = int(h*factor)
                ps_1d_x *= factor # TBD: implement more accurate mapping method rather than nearest neighbor, since the .4 or .6 leads to two different sampling results
                ps_1d_y *= factor                
                
                ps_1d = np.int32(np.floor(ps_1d_x) + np.floor(ps_1d_y) * max_w)
                max_items = max_w * h - 1
                for i in range(len(ps_1d)):
                    if ps_1d[i] > max_items - 1:
                        ps_1d[i] = max_items - 1
                    
                
                image = cv2.resize(image, (max_w, h))
                image = (image-127.5) / 255
            else:
                #print('Warning: {} image not found!'.format(file_path))
                print('{} ignored due to image file not found.'.format(file_path))
                image, ps_1d = None, None
                break
                
            if image is not None and ps_1d is not None: # ignore data with no images                 
                ps_1d_indices.append(ps_1d)
                images.append(image)
                h,w,c = image.shape
                if h > max_h:
                    max_h = h
            else:
                pass
                #print('{} ignored due to image file not found.'.format(file_path))
                
        ## pad image to the same shape
        for i,image in enumerate(images): 
            pad_img = np.zeros([max_h, max_w, 3], dtype=image.dtype)
            pad_img[:image.shape[0], :, :] = image
            images[i] = pad_img
        
        return images, ps_1d_indices
    
    def load_data(self, data_files, update_dict=False):
        """
        label_dressed in format:
        {file_id: {class: [{'key_id':[], 'value_id':[], 'key_text':'', 'value_text':''}, ] } }
        load doc words with location and class returned in format: 
        [[file_name, text, word_id, [x_left, y_top, x_right, y_bottom], [left, top, right, bottom], max_row_words, max_col_words] ]
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
                
                data = self._collect_data(file_id, data['text_boxes'], update_dict)
                for i in data:
                    doc_dressed.append(i)
                    
        return doc_dressed, label_dressed       
    
    def _cal_rows_cols(self, docs, extra_augmentation=False, dropout=False):                  
        max_row = self.encoding_factor
        max_col = self.encoding_factor
        for doc in docs:
            for line in doc: 
                _, _, _, _, _, max_row_words, max_col_words = line
                if max_row_words > max_row:
                    max_row = max_row_words
                if max_col_words > max_col:
                    max_col = max_col_words
        
        pre_rows = self._fit_shape(max_row) #(max_row//self.encoding_factor+1) * self.encoding_factor
        pre_cols = self._fit_shape(max_col) #(max_col//self.encoding_factor+1) * self.encoding_factor
        
        rows, cols = 0, 0
        if extra_augmentation:
            pad_row = int(random.gauss(0, self.da_extra_rows*self.encoding_factor)) #abs(random.gauss(0, u))
            pad_col = int(random.gauss(0, self.da_extra_cols*self.encoding_factor)) #random.randint(0, u)
            
            if self.augment_strategy == 1: # strategy 1: augment data by increasing grid shape sizes
                pad_row = abs(pad_row)
                pad_col = abs(pad_col)
                rows = self._fit_shape(max_row+pad_row) # apply upper boundary to avoid OOM
                cols = self._fit_shape(max_col+pad_col) # apply upper boundary to avoid OOM
            elif self.augment_strategy == 2 or self.augment_strategy == 3: # strategy 2: augment by increasing or decreasing the target gird shape size
                rows = self._fit_shape(max(self.rows_target+pad_row, max_row)) # protect grid shape
                cols = self._fit_shape(max(self.cols_target+pad_col, max_col)) # protect grid shape
            else:
                raise Exception('unknown augment strategy')
            rows = min(rows, self.rows_ulimit) # apply upper boundary to avoid OOM
            cols = min(cols, self.cols_ulimit) # apply upper boundary to avoid OOM                                
        else:
            rows = pre_rows
            cols = pre_cols
        return rows, cols, pre_rows, pre_cols 
    
    def _fit_shape(self, shape): # modify shape size to fit the encoding factor
        while shape % self.encoding_factor:
            shape += 1
        return shape
    
    def _expand_shape(self, shape): # expand shape size with step 2
        return self._fit_shape(shape+1)
        
    def _collect_data(self, file_name, content, update_dict):
        """
        dress and preserve only interested data.
        """          
        content_dressed = []
        left, top, right, bottom, buffer = 9999, 9999, 0, 0, 2
        for line in content:
            bbox = line['bbox'] # handle data corrupt
            if len(bbox) == 0:
                continue
            if line['text'] in self.special_dict: # ignore potential overlap causing characters
                continue
            
            x_left, y_top, x_right, y_bottom = self._dress_bbox(bbox)        
            # TBD: the real image size is better for calculating the relative x/y/w/h
            if x_left < left: left = x_left - buffer
            if y_top < top: top = y_top - buffer
            if x_right > right: right = x_right + buffer
            if y_bottom > bottom: bottom = y_bottom + buffer
            
            word_id = line['id']
            dressed_texts = self._dress_text(line['text'], update_dict)
            
            num_block = len(dressed_texts)
            for i, dressed_text in enumerate(dressed_texts): # handling tokenized text, separate bbox
                new_left = int(x_left + (x_right-x_left) / num_block * (i))
                new_right = int(x_left + (x_right-x_left) / num_block * (i+1))
                content_dressed.append([file_name, dressed_text, word_id, [new_left, y_top, new_right, y_bottom]])
            
        # initial calculation of maximum number of words in rows/cols in terms of image size
        num_words_row = [0 for _ in range(bottom)] # number of words in each row
        num_words_col = [0 for _ in range(right)] # number of words in each column
        for line in content_dressed:
            _, _, _, [x_left, y_top, x_right, y_bottom] = line
            for y in range(y_top, y_bottom):
                num_words_row[y] += 1
            for x in range(x_left, x_right):
                num_words_col[x] += 1
        max_row_words = self._fit_shape(max(num_words_row))
        max_col_words = 0#self._fit_shape(max(num_words_col))
        
        # further expansion of maximum number of words in rows/cols in terms of grid shape
        max_rows = max(self.encoding_factor, max_row_words)
        max_cols = max(self.encoding_factor, max_col_words)
        DONE = False
        while not DONE:
            DONE = True
            grid_table = np.zeros([max_rows, max_cols], dtype=np.int32)
            for line in content_dressed:
                _, _, _, [x_left, y_top, x_right, y_bottom] = line
                row = int(max_rows * (y_top - top + (y_bottom-y_top)/2) / (bottom-top))
                col = int(max_cols * (x_left - left + (x_right-x_left)/2) / (right-left))
                #row = int(max_rows * (y_top + (y_bottom-y_top)/2) / (bottom))
                #col = int(max_cols * (x_left + (x_right-x_left)/2) / (right))
                #row = int(max_rows * (y_top-top) / (bottom-top))
                #col = int(max_cols * (x_left-left) / (right-left))
                #row = int(max_rows * (y_top) / (bottom))
                #col = int(max_cols * (x_left) / (right))
                #row = int(max_rows * (y_top + (y_bottom-y_top)/2) / bottom)  
                #col = int(max_cols * (x_left + (x_right-x_left)/2) / right) 
                
                while col < max_cols and grid_table[row, col] != 0: # shift to find slot to drop the current item
                    col += 1
                if col == max_cols: # shift to find slot to drop the current item
                    col -= 1
                    ptr = 0
                    while ptr<max_cols and grid_table[row, ptr] != 0:
                        ptr += 1
                    if ptr == max_cols: # overlap cannot be solved in current row, then expand the grid
                        max_cols = self._expand_shape(max_cols)
                        DONE = False
                        break
                    
                    grid_table[row, ptr:-1] = grid_table[row, ptr+1:]
                
                if DONE:
                    if row > max_rows or col>max_cols:
                        print('wrong')
                    grid_table[row, col] = 1
        
        max_rows = self._fit_shape(max_rows)
        max_cols = self._fit_shape(max_cols)
        
        #print('{} collected in shape: {},{}'.format(file_name, max_rows, max_cols))
        
        # segment grid into two parts if number of cols is larger than self.cols_target
        data = []
        if self.segment_grid and max_cols > self.cols_segment:
            content_dressed_left = []
            content_dressed_right = []
            cnt = defaultdict(int) # counter for number of words in a specific row
            cnt_l, cnt_r = defaultdict(int), defaultdict(int) # update max_cols if larger than self.cols_segment
            left_boundary = max_cols - self.cols_segment
            right_boundary = self.cols_segment
            for i, line in enumerate(content_dressed):
                file_name, dressed_text, word_id, [x_left, y_top, x_right, y_bottom] = line
                
                row = int(max_rows * (y_top + (y_bottom-y_top)/2) / bottom)
                cnt[row] += 1                
                if cnt[row] <= left_boundary:
                    cnt_l[row] += 1
                    content_dressed_left.append([file_name, dressed_text, word_id, [x_left, y_top, x_right, y_bottom], \
                                      [left, top, right, bottom], max_rows, self.cols_segment])
                elif left_boundary < cnt[row] <= right_boundary:
                    cnt_l[row] += 1
                    cnt_r[row] += 1
                    content_dressed_left.append([file_name, dressed_text, word_id, [x_left, y_top, x_right, y_bottom], \
                                      [left, top, right, bottom], max_rows, self.cols_segment])
                    content_dressed_right.append([file_name, dressed_text, word_id, [x_left, y_top, x_right, y_bottom], \
                                      [left, top, right, bottom], max_rows, max(max(cnt_r.values()), self.cols_segment)])
                else:
                    cnt_r[row] += 1
                    content_dressed_right.append([file_name, dressed_text, word_id, [x_left, y_top, x_right, y_bottom], \
                                      [left, top, right, bottom], max_rows, max(max(cnt_r.values()), self.cols_segment)])
            #print(sorted(cnt.items(), key=lambda x:x[1], reverse=True))
            #print(sorted(cnt_l.items(), key=lambda x:x[1], reverse=True))
            #print(sorted(cnt_r.items(), key=lambda x:x[1], reverse=True))
            if max(cnt_l.values()) < 2*self.cols_segment:
                data.append(content_dressed_left)
            if max(cnt_r.values()) < 2*self.cols_segment: # avoid OOM, which tends to happen in the right side
                data.append(content_dressed_right)
        else:
            for i, line in enumerate(content_dressed): # append height/width/numofwords to the list
                file_name, dressed_text, word_id, [x_left, y_top, x_right, y_bottom] = line
                content_dressed[i] = [file_name, dressed_text, word_id, [x_left, y_top, x_right, y_bottom], \
                                      [left, top, right, bottom], max_rows, max_cols ]
            data.append(content_dressed)
        return data
    
    def _collect_label(self, file_id, content):
        """
        dress and preserve only interested data.
        label_dressed in format:
        {file_id: {class: [{'key_id':[], 'value_id':[], 'key_text':'', 'value_text':''}, ] } }
        """
        label_dressed = dict()
        label_dressed[file_id] = {cls:[] for cls in self.classes[1:]}
        for line in content:
            cls = line['field_name']
            if cls in self.classes:
                #identity = line.get('identity', 0) 
                label_dressed[file_id][cls].append( {'key_id':[], 'value_id':[], 'key_text':'', 'value_text':''} )
                label_dressed[file_id][cls][-1]['key_id'] = line.get('key_id', [])
                label_dressed[file_id][cls][-1]['value_id'] = line['value_id'] # value_id
                label_dressed[file_id][cls][-1]['key_text'] = line.get('key_text', []) 
                label_dressed[file_id][cls][-1]['value_text'] = line['value_text'] # value_text
                
        # handle corrupted data
        for cls in label_dressed[file_id]: 
            for idx, label in enumerate(label_dressed[file_id][cls]):
                if len(label) == 0: # no relevant class in sample @file_id
                    continue
                if (len(label['key_text'])>0 and len(label['key_id'])==0) or \
                   (len(label['value_text'])>0 and len(label['value_id'])==0):
                    return None
            
        return label_dressed

    def _dress_class(self, file_name, word_id, labels):
        """
        label_dressed in format:
        {file_id: {class: [{'key_id':[], 'value_id':[], 'key_text':'', 'value_text':''}, ] } }
        """
        if file_name in labels:
            for cls, cls_labels in labels[file_name].items():
                for idx, cls_label in enumerate(cls_labels):
                    for key, values in cls_label.items():
                        if (key=='key_id' or key=='value_id') and word_id in values:
                            if key == 'key_id':
                                if self.data_mode == 0:
                                    return idx, self.classes.index(cls) * 2 - 1 # odd
                                elif self.data_mode == 1:
                                    return idx, self.classes.index(cls)
                                else: # ignore key_id when self.data_mode is not 0 or 1
                                    return 0, 0
                            elif key == 'value_id':
                                if self.data_mode == 0:
                                    return idx, self.classes.index(cls) * 2 # even 
                                else: # when self.data_mode is 1 or 2
                                    return idx, self.classes.index(cls) 
            return 0, 0 # 0 is of class type 'DontCare'
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
                    #print('unknown text: ' + string)
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
            
    def grid_visualization(self, file_name, grid, label):
        import cv2
        height, width = np.shape(grid)
        grid_box_h, grid_box_w = 20, 40
        palette = np.zeros([height*grid_box_h, width*grid_box_w, 3], np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        gt_color = [[255, 250, 240], [152, 245, 255], [127, 255, 212], [100, 149, 237], 
                    [192, 255, 62], [175, 238, 238], [255, 130, 171], [240, 128, 128], [255, 105, 180]]
        cv2.putText(palette, file_name+"({},{})".format(height,width), (grid_box_h,grid_box_w), font, 0.6, [255,0,0])  
        for h in range(height):
            cv2.line(palette, (0,h*grid_box_h), (width*grid_box_w, h*grid_box_h), (100,100,100))
            for w in range(width):
                if grid[h,w]:
                    org = (int((w+1)*grid_box_w*0.7),int((h+1)*grid_box_h*0.9))
                    color = gt_color[label[h,w]]
                    cv2.putText(palette, self.index_to_word[grid[h,w]], org, font, 0.4, color)        
        
        img = cv2.imread(self.doc_path+'/'+file_name)
        if img is not None:
            shape = list(img.shape)
            max_len = 768
            factor = max_len / max(shape)
            shape[0], shape[1] = [int(s*factor) for s in shape[:2]]
            img = cv2.resize(img, (shape[1], shape[0]))  
            cv2.imshow("img", img)
        cv2.imshow("grid", palette)
        cv2.waitKey(0)