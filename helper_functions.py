import numpy as np
import jiwer

def get_xy_split_data(input_data, output_data, split=0.2):
    '''
    Helper function to get and split X, Y data
    '''
    split_in = int(split*len(input_data))
    split_out = int(split*len(output_data))
    X_train, y_train = input_data[:-split_in], output_data[:-split_out]
    X_test, y_test = input_data[-split_in:], output_data[-split_out:]
    
    return X_train, y_train, X_test, y_test

def flat_n_get_string(alist):
    '''
    Helper function for WER calculation
    Flatten a list of list with a label as an item
    Find unique labels
    and return a string of concanated labels
    '''
    # flatten a list of list and find unique labels
    string_set = set()
    for i in np.array(alist).flatten(): string_set.add(i)

    # concat labels into a string
    astring = ""
    for word in string_set:
        astring += (word + " ")

    return astring

def decode_levenshtein(p, t):
    '''
    Decode the calculation of WER - to get the Substitution, Deletions, Insertions, Equals
    Arguments: 
    - p = hypothesis (predictions), type: str
    - t = references (targets), type: str
    Return: 
    - List[List[type, predict_phone, reference_phone]]

    Notes:
    * Output of jiwer.process_words 
        - A WordOutput instance that has an attribute "alignments", which is in type of List[List[AlignmentChunk]]
        - link to documentation: https://jitsi.github.io/jiwer/reference/process/#process.AlignmentChunk
    '''
    ops = []
    p_lst = p.split()
    t_lst = t.split()

    out = jiwer.process_words(t, p) # get WordOutput instance
    for a in out.alignments[0]:     # get AlignmentChunk
        op = a.type
        if op == "substitute":
            pred_ix = a.hyp_start_idx
            targ_ix = a.ref_start_idx
            ops.append([op, p_lst[pred_ix], t_lst[targ_ix]])
        if op == "insert":
            pred_ix = a.hyp_start_idx
            ops.append([op, p_lst[pred_ix], p_lst[pred_ix]])
        if op == "delete":
            targ_ix = a.ref_start_idx
            ops.append([op, t_lst[targ_ix], t_lst[targ_ix]])
        if op == "equal":
            pred_ix = a.hyp_start_idx
            targ_ix = a.ref_start_idx
            ops.append([op, p_lst[pred_ix], t_lst[targ_ix]])

    return ops


