# written by Xiaohui Zhao
# 2018-01 
# xiaohui.zhao@accenture.com
import numpy as np
from os.path import join
try:
    import cv2
except ImportError:
    pass

c_threshold = 0.5

def cal_accuracy(data_loader, grid_table, gt_classes, model_output_val, label_mapids, bbox_mapids):
    #num_tp = 0
    #num_fn = 0
    res = ''
    num_correct = 0
    num_correct_strict = 0
    num_correct_soft = 0
    num_all = grid_table.shape[0] * (model_output_val.shape[-1]-1)
    for b in range(grid_table.shape[0]):
        data_input_flat = grid_table[b,:,:,0].reshape([-1])
        labels = gt_classes[b,:,:].reshape([-1])
        logits = model_output_val[b,:,:,:].reshape([-1, data_loader.num_classes])
        label_mapid = label_mapids[b]
        bbox_mapid = bbox_mapids[b]
        rows, cols = grid_table.shape[1:3]
        bbox_id = np.array([row*cols+col for row in range(rows) for col in range(cols)])
        
        # ignore inputs that are not word
        indexes = np.where(data_input_flat != 0)[0]
        data_selected = data_input_flat[indexes]
        labels_selected = labels[indexes]
        logits_array_selected = logits[indexes]
        bbox_id_selected = bbox_id[indexes]
        
        # calculate accuracy
        #test_classes = [1,2,3,4,5,6,7,8,9]
        for c in range(1, data_loader.num_classes):
            labels_indexes = np.where(labels_selected == c)[0]
            logits_indexes = np.where(logits_array_selected[:,c] > c_threshold)[0]
            
            labels_words = list(data_loader.index_to_word[i] for i in data_selected[labels_indexes])
            logits_words = list(data_loader.index_to_word[i] for i in data_selected[logits_indexes])
            
            label_bbox_ids = label_mapid[c] # GT bbox_ids related to the type of class
            logit_bbox_ids = [bbox_mapid[bbox] for bbox in bbox_id_selected[logits_indexes] if bbox in bbox_mapid]            
            
            #if np.array_equal(labels_indexes, logits_indexes):
            if set(label_bbox_ids) == set(logit_bbox_ids): # decide as correct when all ids match
                num_correct_strict += 1  
                num_correct_soft += 1
            elif set(label_bbox_ids).issubset(set(logit_bbox_ids)): # correct when gt is subset of gt
                num_correct_soft += 1
            try: # calculate prevalence with decimal precision
                num_correct += np.shape(np.intersect1d(labels_indexes, logits_indexes))[0] / np.shape(labels_indexes)[0]
            except ZeroDivisionError:
                if np.shape(labels_indexes)[0] == 0:
                    num_correct += 1
                else:
                    num_correct += 0        
            
            # show results without the <DontCare> class                    
            if b==0:
                res += '\n{}(GT/Inf):\t"'.format(data_loader.classes[c])
                
                # ground truth label
                res += ' '.join(data_loader.index_to_word[i] for i in data_selected[labels_indexes])
                res += '" | "'
                res += ' '.join(data_loader.index_to_word[i] for i in data_selected[logits_indexes])
                res += '"'
                
                # wrong inferences results
                if not np.array_equal(labels_indexes, logits_indexes): 
                    res += '\n \t FALSES =>>'
                    logits_flat = logits_array_selected[:,c]
                    fault_logits_indexes = np.setdiff1d(logits_indexes, labels_indexes)
                    for i in range(len(data_selected)):
                        if i not in fault_logits_indexes: # only show fault_logits_indexes
                            continue
                        w = data_loader.index_to_word[data_selected[i]]
                        l = data_loader.classes[labels_selected[i]]
                        res += ' "%s"/%s, '%(w, l)
                        #res += ' "%s"/%.2f%s, '%(w, logits_flat[i], l)
                        
                #print(res)
    prevalence = num_correct / num_all
    accuracy_strict = num_correct_strict / num_all
    accuracy_soft = num_correct_soft / num_all
    return prevalence, accuracy_strict, accuracy_soft, res.encode("utf-8")


def cal_save_results(data_loader, save_prefix, docs, grid_table, gt_classes, model_output_val):
    res = ''
    num_correct = 0
    num_correct_strict = 0
    num_all = grid_table.shape[0] * (model_output_val.shape[-1]-1)
    for b in range(grid_table.shape[0]):
        filename = docs[b][0][0]
        
        data_input_flat = grid_table[b,:,:,0].reshape([-1])
        labels = gt_classes[b,:,:].reshape([-1])
        logits = model_output_val[b,:,:,:].reshape([-1, data_loader.num_classes])
        
        # ignore inputs that are not word
        indexes = np.where(data_input_flat != 0)[0]
        data_selected = data_input_flat[indexes]
        labels_selected = labels[indexes]
        logits_array_selected = logits[indexes]
        
        # calculate accuracy
        for c in range(1, data_loader.num_classes):
            labels_indexes = np.where(labels_selected == c)[0]
            logits_indexes = np.where(logits_array_selected[:,c] > c_threshold)[0]
            if np.array_equal(labels_indexes, logits_indexes): 
                num_correct_strict += 1        
            try:  
                num_correct += np.shape(np.intersect1d(labels_indexes, logits_indexes))[0] / np.shape(labels_indexes)[0]
            except ZeroDivisionError:
                if np.shape(logits_indexes)[0] == 0:
                    num_correct += 1
                else:
                    num_correct += 0                    
        
            # show results without the <DontCare> class      
            res += '\n{}(GT/Inf):\t"'.format(data_loader.classes[c])
            
            # ground truth label
            gt = str(' '.join([data_loader.index_to_word[i] for i in data_selected[labels_indexes]]).encode('utf-8'))
            predict = str(' '.join([data_loader.index_to_word[i] for i in data_selected[logits_indexes]]).encode('utf-8'))
            res += gt + '" | "' + predict + '"'
        
            # write results to csv
            fieldnames = ['TaskID', 'GT', 'Predicted']
            
            csv_filename = 'data/results/' + save_prefix + '_' + data_loader.classes[c] + '.csv'            
            writer = csv.DictWriter(open(csv_filename, 'a'), fieldnames=fieldnames) 
            row = {'TaskID':filename, 'GT':gt, 'Predicted':predict}
            writer.writerow(row)
            
            csv_diff_filename = 'data/results/' + save_prefix + '_Diff_' + data_loader.classes[c] + '.csv'
            if gt != predict:
                writer = csv.DictWriter(open(csv_diff_filename, 'a'), fieldnames=fieldnames) 
                row = {'TaskID':filename, 'GT':gt, 'Predicted':predict}
                writer.writerow(row)
            
            # wrong inferences results
            if not np.array_equal(labels_indexes, logits_indexes): 
                res += '\n \t FALSES =>>'
                logits_flat = logits_array_selected[:,c]
                fault_logits_indexes = np.setdiff1d(logits_indexes, labels_indexes)
                for i in range(len(data_selected)):
                    if i not in fault_logits_indexes: # only show fault_logits_indexes
                        continue
                    w = data_loader.index_to_word[data_selected[i]]
                    l = data_loader.classes[labels_selected[i]]
                    res += ' "%s"/%s, '%(w, l)
                    #res += ' "%s"/%.2f%s, '%(w, logits_flat[i], l)
                        
            #print(res)
    recall = num_correct / num_all
    accuracy_strict = num_correct_strict / num_all
    return recall, accuracy_strict, res


def vis_bbox(data_loader, file_prefix, grid_table, gt_classes, model_output_val, file_name, bboxes, shape):
    data_input_flat = grid_table.reshape([-1])
    labels = gt_classes.reshape([-1])
    logits = model_output_val.reshape([-1, data_loader.num_classes])
    bboxes = bboxes.reshape([-1])
    
    max_len = 768*2 # upper boundary of image display size 
    img = cv2.imread(join(file_prefix, file_name))
    if img is not None:    
        shape = list(img.shape)
        
        bbox_pad = 1
        gt_color = [[255, 250, 240], [152, 245, 255], [119,204,119], [100, 149, 237], 
                    [192, 255, 62], [119,119,204], [114,124,114], [240, 128, 128], [255, 105, 180]]
        inf_color = [[255, 222, 173], [0, 255, 255], [50,219,50], [72, 61, 139], 
                     [154, 205, 50], [50,50,219], [64,76,64], [255, 0, 0], [255, 20, 147]]
        
        font_size = 0.5
        font = cv2.FONT_HERSHEY_COMPLEX
        ft_color = [50, 50, 250]
        
        factor = max_len / max(shape)
        shape[0], shape[1] = [int(s*factor) for s in shape[:2]]
        
        img = cv2.resize(img, (shape[1], shape[0]))        
        overlay_box = np.zeros(shape, dtype=img.dtype)
        overlay_line = np.zeros(shape, dtype=img.dtype)
        for i in range(len(data_input_flat)):
            if len(bboxes[i]) > 0:
                x,y,w,h = [int(p*factor) for p in bboxes[i]]
            else:
                row = i // data_loader.rows
                col = i % data_loader.cols
                x = shape[1] // data_loader.cols * col
                y = shape[0] // data_loader.rows * row
                w = shape[1] // data_loader.cols * 2
                h = shape[0] // data_loader.cols * 2
                
            if data_input_flat[i] and labels[i]:
                gt_id = labels[i]                
                cv2.rectangle(overlay_box, (x,y), (x+w,y+h), gt_color[gt_id], -1)
                    
            if max(logits[i]) > c_threshold:
                inf_id = np.argmax(logits[i])
                if inf_id:                
                    cv2.rectangle(overlay_line, (x+bbox_pad,y+bbox_pad), \
                                  (x+bbox_pad+w,y+bbox_pad+h), inf_color[inf_id], max_len//768*2)
                
            #text = data_loader.classes[gt_id] + '|' + data_loader.classes[inf_id]
            #cv2.putText(img, text, (x,y), font, font_size, ft_color)  
        
        # legends
        w = shape[1] // data_loader.cols * 4
        h = shape[0] // data_loader.cols * 2
        for i in range(1, len(data_loader.classes)):
            row = i * 3
            col = 0
            x = shape[1] // data_loader.cols * col
            y = shape[0] // data_loader.rows * row 
            cv2.rectangle(img, (x,y), (x+w,y+h), gt_color[i], -1)
            cv2.putText(img, data_loader.classes[i], (x+w,y+h), font, 0.8, ft_color)  
            
            row = i * 3 + 1
            col = 0
            x = shape[1] // data_loader.cols * col
            y = shape[0] // data_loader.rows * row 
            cv2.rectangle(img, (x+bbox_pad,y+bbox_pad), \
                          (x+bbox_pad+w,y+bbox_pad+h), inf_color[i], max_len//384)        
        
        alpha = 0.4
        cv2.addWeighted(overlay_box, alpha, img, 1-alpha, 0, img)
        cv2.addWeighted(overlay_line, 1-alpha, img, 1, 0, img)
        cv2.imwrite('results/' + file_name[:-4]+'.png', img)        
        cv2.imshow("test", img)
        cv2.waitKey(0)