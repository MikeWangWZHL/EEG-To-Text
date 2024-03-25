import numpy as np
import re

eeg_float_resolution=np.float16

Alpha_ffd_names = ['FFD_a1', 'FFD_a1_diff', 'FFD_a2', 'FFD_a2_diff']
Beta_ffd_names = ['FFD_b1', 'FFD_b1_diff', 'FFD_b2', 'FFD_b2_diff']
Gamma_ffd_names = ['FFD_g1', 'FFD_g1_diff', 'FFD_g2', 'FFD_g2_diff']
Theta_ffd_names = ['FFD_t1', 'FFD_t1_diff', 'FFD_t2', 'FFD_t2_diff']
Alpha_gd_names = ['GD_a1', 'GD_a1_diff', 'GD_a2', 'GD_a2_diff']
Beta_gd_names = ['GD_b1', 'GD_b1_diff', 'GD_b2', 'GD_b2_diff']
Gamma_gd_names = ['GD_g1', 'GD_g1_diff', 'GD_g2', 'GD_g2_diff']
Theta_gd_names = ['GD_t1', 'GD_t1_diff', 'GD_t2', 'GD_t2_diff']
Alpha_gpt_names = ['GPT_a1', 'GPT_a1_diff', 'GPT_a2', 'GPT_a2_diff']
Beta_gpt_names = ['GPT_b1', 'GPT_b1_diff', 'GPT_b2', 'GPT_b2_diff']
Gamma_gpt_names = ['GPT_g1', 'GPT_g1_diff', 'GPT_g2', 'GPT_g2_diff']
Theta_gpt_names = ['GPT_t1', 'GPT_t1_diff', 'GPT_t2', 'GPT_t2_diff']
Alpha_sfd_names = ['SFD_a1', 'SFD_a1_diff', 'SFD_a2', 'SFD_a2_diff']
Beta_sfd_names = ['SFD_b1', 'SFD_b1_diff', 'SFD_b2', 'SFD_b2_diff']
Gamma_sfd_names = ['SFD_g1', 'SFD_g1_diff', 'SFD_g2', 'SFD_g2_diff']
Theta_sfd_names = ['SFD_t1', 'SFD_t1_diff', 'SFD_t2', 'SFD_t2_diff']
Alpha_trt_names = ['TRT_a1', 'TRT_a1_diff', 'TRT_a2', 'TRT_a2_diff']
Beta_trt_names = ['TRT_b1', 'TRT_b1_diff', 'TRT_b2', 'TRT_b2_diff']
Gamma_trt_names = ['TRT_g1', 'TRT_g1_diff', 'TRT_g2', 'TRT_g2_diff']
Theta_trt_names = ['TRT_t1', 'TRT_t1_diff', 'TRT_t2', 'TRT_t2_diff']

# IF YOU CHANGE THOSE YOU MUST ALSO CHANGE CONSTANTS
Alpha_features = Alpha_ffd_names + Alpha_gd_names + Alpha_gpt_names + Alpha_trt_names# + Alpha_sfd_names
Beta_features = Beta_ffd_names + Beta_gd_names + Beta_gpt_names + Beta_trt_names# + Beta_sfd_names
Gamma_features = Gamma_ffd_names + Gamma_gd_names + Gamma_gpt_names + Gamma_trt_names# + Gamma_sfd_names
Theta_features = Theta_ffd_names + Theta_gd_names + Theta_gpt_names + Theta_trt_names# + Theta_sfd_names
# print(Alpha_features)

# GD_EEG_feautres


def extract_all_fixations(data_container, word_data_object, float_resolution = np.float16):
    """
    Extracts all fixations from a word data object
    :param data_container:      (h5py)  Container of the whole data, h5py object
    :param word_data_object:    (h5py)  Container of fixation objects, h5py object
    :param float_resolution:    (type)  Resolution to which data re to be converted, used for data compression
    :return:
        fixations_data  (list)  Data arrays representing each fixation
    """
    word_data = data_container[word_data_object]
    fixations_data = []
    if len(word_data.shape) > 1:
        for fixation_idx in range(word_data.shape[0]):
            fixations_data.append(np.array(data_container[word_data[fixation_idx][0]]).astype(float_resolution))
    return fixations_data


def is_real_word(word):
    """
    Check if the word is a real word
    :param word:    (str)   word string
    :return:
        is_word (bool)  True if it is a real word
    """
    is_word = re.search('[a-zA-Z0-9]', word)
    return is_word


def load_matlab_string(matlab_extracted_object):
    """
    Converts a string loaded from h5py into a python string
    :param matlab_extracted_object:     (h5py)  matlab string object
    :return:
        extracted_string    (str)   translated string
    """
    extracted_string = u''.join(chr(c[0]) for c in matlab_extracted_object)
    return extracted_string


def extract_word_level_data(data_container, word_objects, eeg_float_resolution = np.float16):
    """
    Extracts word level data for a specific sentence
    :param data_container:          (h5py)  Container of the whole data, h5py object
    :param word_objects:            (h5py)  Container of all word data for a specific sentence
    :param eeg_float_resolution:    (type)  Resolution with which to save EEG, used for data compression
    :return:
        word_level_data     (dict)  Contains all word level data indexed by their index number in the sentence,
                                    together with the reading order, indexed by "word_reading_order"
    """
    available_objects = list(word_objects)
    #print(available_objects)
    #print(len(available_objects))
    # print('available_objects:', available_objects)

    if isinstance(available_objects[0], str):

        contentData = word_objects['content']
        #fixations_order_per_word = []
        if "rawEEG" in available_objects:

            rawData = word_objects['rawEEG']
            etData = word_objects['rawET']

            ffdData = word_objects['FFD']
            gdData = word_objects['GD']
            gptData = word_objects['GPT']
            trtData = word_objects['TRT']

            try:
                sfdData = word_objects['SFD']
            except KeyError:
                print("no SFD!")
                sfdData = []
            nFixData = word_objects['nFixations']
            fixPositions = word_objects["fixPositions"]

            Alpha_features_data = [word_objects[feature] for feature in Alpha_features]
            Beta_features_data = [word_objects[feature] for feature in Beta_features]
            Gamma_features_data = [word_objects[feature] for feature in Gamma_features]
            Theta_features_data = [word_objects[feature] for feature in Theta_features]
            #### 
            GD_EEG_features = [word_objects[feature] for feature in ['GD_t1','GD_t2','GD_a1','GD_a2','GD_b1','GD_b2','GD_g1','GD_g2']]
            FFD_EEG_features = [word_objects[feature] for feature in ['FFD_t1','FFD_t2','FFD_a1','FFD_a2','FFD_b1','FFD_b2','FFD_g1','FFD_g2']]
            TRT_EEG_features = [word_objects[feature] for feature in ['TRT_t1','TRT_t2','TRT_a1','TRT_a2','TRT_b1','TRT_b2','TRT_g1','TRT_g2']]
            #### 
            assert len(contentData) == len(etData) == len(rawData), "different amounts of different data!!"

            zipped_data = zip(rawData, etData, contentData, ffdData, gdData, gptData, trtData, sfdData, nFixData, fixPositions)
            
            word_level_data = {}
            word_idx = 0

            word_tokens_has_fixation = [] 
            word_tokens_with_mask = []
            word_tokens_all = []
            for raw_eegs_obj, ets_obj, word_obj, ffd, gd, gpt, trt, sfd, nFix, fixPos in zipped_data:
                word_string = load_matlab_string(data_container[word_obj[0]])
                if is_real_word(word_string):
                    data_dict = {}
                    data_dict["RAW_EEG"] = extract_all_fixations(data_container, raw_eegs_obj[0], eeg_float_resolution)
                    data_dict["RAW_ET"] = extract_all_fixations(data_container, ets_obj[0], np.float32)

                    data_dict["FFD"] = data_container[ffd[0]][()][0, 0] if len(data_container[ffd[0]][()].shape) == 2 else None
                    data_dict["GD"] = data_container[gd[0]][()][0, 0] if len(data_container[gd[0]][()].shape) == 2 else None
                    data_dict["GPT"] = data_container[gpt[0]][()][0, 0] if len(data_container[gpt[0]][()].shape) == 2 else None
                    data_dict["TRT"] = data_container[trt[0]][()][0, 0] if len(data_container[trt[0]][()].shape) == 2 else None
                    data_dict["SFD"] = data_container[sfd[0]][()][0, 0] if len(data_container[sfd[0]][()].shape) == 2 else None
                    data_dict["nFix"] = data_container[nFix[0]][()][0, 0] if len(data_container[nFix[0]][()].shape) == 2 else None

                    #fixations_order_per_word.append(np.array(data_container[fixPos[0]]))

                    #print([data_container[obj[word_idx][0]][()] for obj in Alpha_features_data])


                    data_dict["ALPHA_EEG"] = np.concatenate([data_container[obj[word_idx][0]][()]
                                                             if len(data_container[obj[word_idx][0]][()].shape) == 2 else []
                                                             for obj in Alpha_features_data], 0)

                    data_dict["BETA_EEG"] = np.concatenate([data_container[obj[word_idx][0]][()]
                                                            if len(data_container[obj[word_idx][0]][()].shape) == 2 else []
                                                            for obj in Beta_features_data], 0)

                    data_dict["GAMMA_EEG"] = np.concatenate([data_container[obj[word_idx][0]][()]
                                                             if len(data_container[obj[word_idx][0]][()].shape) == 2 else []
                                                             for obj in Gamma_features_data], 0)

                    data_dict["THETA_EEG"] = np.concatenate([data_container[obj[word_idx][0]][()]
                                                             if len(data_container[obj[word_idx][0]][()].shape) == 2 else []
                                                             for obj in Theta_features_data], 0)




                    data_dict["word_idx"] = word_idx
                    data_dict["content"] = word_string
                    ####################################
                    word_tokens_all.append(word_string)
                    if data_dict["nFix"] is not None:
                        ####################################
                        data_dict["GD_EEG"] = [np.squeeze(data_container[obj[word_idx][0]][()]) if len(data_container[obj[word_idx][0]][()].shape) == 2 else [] for obj in GD_EEG_features]
                        data_dict["FFD_EEG"] = [np.squeeze(data_container[obj[word_idx][0]][()]) if len(data_container[obj[word_idx][0]][()].shape) == 2 else [] for obj in FFD_EEG_features]
                        data_dict["TRT_EEG"] = [np.squeeze(data_container[obj[word_idx][0]][()]) if len(data_container[obj[word_idx][0]][()].shape) == 2 else [] for obj in TRT_EEG_features]
                        ####################################
                        word_tokens_has_fixation.append(word_string)
                        word_tokens_with_mask.append(word_string)
                    else:
                        word_tokens_with_mask.append('[MASK]')
                        

                    word_level_data[word_idx] = data_dict
                    word_idx += 1
                else:
                    print(word_string + " is not a real word.")
        else:
            # If there are no word-level data it will be word embeddings alone
            word_level_data = {}
            word_idx = 0
            word_tokens_has_fixation = [] 
            word_tokens_with_mask = []
            word_tokens_all = []

            for word_obj in contentData:
                word_string = load_matlab_string(data_container[word_obj[0]])
                if is_real_word(word_string):
                    data_dict = {}
                    data_dict["RAW_EEG"] = []
                    data_dict["ICA_EEG"] = []
                    data_dict["RAW_ET"] = []
                    data_dict["FFD"] = None
                    data_dict["GD"] = None
                    data_dict["GPT"] = None
                    data_dict["TRT"] = None
                    data_dict["SFD"] = None
                    data_dict["nFix"] = None
                    data_dict["ALPHA_EEG"] = []
                    data_dict["BETA_EEG"] = []
                    data_dict["GAMMA_EEG"] = []
                    data_dict["THETA_EEG"] = []

                    data_dict["word_idx"] = word_idx
                    data_dict["content"] = word_string
                    word_level_data[word_idx] = data_dict
                    word_idx += 1
                else:
                    print(word_string + " is not a real word.")

            sentence = " ".join([load_matlab_string(data_container[word_obj[0]]) for word_obj in word_objects['content']])
            #print("Only available objects for the sentence '{}' are {}.".format(sentence, available_objects))
            #word_level_data["word_reading_order"] = extract_word_order_from_fixations(fixations_order_per_word)
    else:
        word_tokens_has_fixation = [] 
        word_tokens_with_mask = []
        word_tokens_all = []
        word_level_data = {}
    return word_level_data, word_tokens_all, word_tokens_has_fixation, word_tokens_with_mask