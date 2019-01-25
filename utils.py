# written by Xiaohui Zhao
# 2018-01 
# xiaohui.zhao@accenture.com
import numpy as np

def cal_accuracy(data_loader, gird_table, gt_classes, model_output_val, c_threshold):
    #num_tp = 0
    #num_fn = 0
    res = ''
    num_correct = 0
    num_correct_strict = 0
    num_all = gird_table.shape[0] * (model_output_val.shape[-1]-1)
    for b in range(gird_table.shape[0]):
        data_input_flat = gird_table[b,:,:,0].reshape([-1])
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
            if b==0:
                res += '\n{}(GT/Inf):\t"'.format(data_loader.classes[c])
                
                # ground truth label
                res += ' '.join(data_loader.index_to_word[i] for i in data_selected[labels_indexes])
                res += '" | "'
                res += ' '.join(data_loader.index_to_word[i] for i in data_selected[logits_indexes])
                res += '"'
                
                # wrong inferences results
                if not np.array_equal(labels_indexes, logits_indexes): 
                    res += '"\n \t FALSES =>>'
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

def calc_ghm_weights(logits, labels, acc_sum, bins, momentum): 
    """
    calculate gradient harmonizing mechanism weights
    """
    bins = bins
    shape = logits.shape
    edges = [float(x)/bins for x in range(bins+1)]
    edges[-1] += 1e-6 
    
    logits_flat = logits.reshape([-1])
    labels_flat = labels.reshape([-1])
    arr = []
    for l in labels_flat:
        arr.extend([i==l for i in range(shape[-1])]) 
    labels_flat = np.array(arr)
    
    grad = abs(logits_flat - labels_flat) # equation for logits from the sigmoid activation
    
    weights = np.ones(logits_flat.shape)
    N = shape[0] * shape[1] * shape[2] * shape[3]
    M = 0
    for i in range(bins):
        idxes = np.multiply(grad>=edges[i], grad<edges[i+1])
        num_in_bin = np.sum(idxes)
        if num_in_bin > 0: 
            acc_sum[i] = momentum * acc_sum[i] + (1-momentum) * num_in_bin
            weights[np.where(idxes)] = N / acc_sum[i]
            M += 1
    if M > 0:
        weights = weights / M
        
    return weights.reshape(shape)
    