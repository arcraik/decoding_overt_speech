import numpy as np
import mne

def get_identifiers(cleaning_types, base_folder, subject_name, test_option):

    identifiers = []
    processing_selections = []
    for pp in range(0, len(cleaning_types)):
        cleaning_type = cleaning_types[pp]

        if cleaning_type == 'no_clean':
            processing_selection = '/EEG_Preprocessed_test'
            identifiers = np.append(identifiers, '')
            processing_selections = np.append(processing_selections, processing_selection)
        else:
            identifier_location = base_folder + subject_name + "/EEG_Cleaned" + test_option + "/" + cleaning_type + "/" + cleaning_type + "_identifiers.csv"
            temp_identifier = np.loadtxt(identifier_location, delimiter=",", dtype=str)
            identifiers = np.append(identifiers, temp_identifier)

            processing_selection = '/EEG_Cleaned' + test_option + '/' + cleaning_type
            temp_processing_selection = np.array([processing_selection for _ in range(len(temp_identifier))])
            processing_selections = np.append(processing_selections, temp_processing_selection)

    return identifiers, processing_selections

def get_block_labels(label_indices, channels, window_size, block_events, label_inclusion, labels_dictionary, class_selection, no_overlap, EEG_data, OFFSET):

    # Initialize block data and labels arrays
    temp_block_data = np.zeros((len(label_indices), channels, int(window_size)))
    temp_block_labels = np.zeros((len(label_indices), 1))
    count = 0
    previous_sample = 0

    for i in range(len(label_indices)):

        if block_events[label_indices[i]] != "":

            full_temp_label = block_events[label_indices[i]]

            if bool([ele for ele in label_inclusion if (ele in full_temp_label)]):
                temp_label = full_temp_label[0:-2]

                temp_class = int(labels_dictionary[temp_label][class_selection])

                if no_overlap == 1:
                    if temp_class != 999 and (label_indices[i] - previous_sample >= window_size / 2):
                        temp_EEG = np.array(EEG_data[:, int(label_indices[i] - (window_size / 2) - OFFSET):int(
                            label_indices[i] + (window_size / 2) - OFFSET)])
                        temp_block_data[count] = temp_EEG
                        temp_block_labels[count] = int(labels_dictionary[temp_label][class_selection])
                        previous_sample = label_indices[i]
                        count += 1
                else:
                    if temp_class != 999:
                        temp_EEG = np.array(EEG_data[:, int(label_indices[i] - (window_size / 2) - OFFSET):int(
                            label_indices[i] + (window_size / 2) - OFFSET)])
                        temp_block_data[count] = temp_EEG
                        temp_block_labels[count] = int(labels_dictionary[temp_label][class_selection])
                        count += 1
    return temp_block_labels, temp_block_data, count

def get_identifier_data(channels, window_size, max_sessions, max_blocks, base_file_path, subject_name, processing_selection, identifier, filter_data, sfreq, l_freq, h_freq, label_inclusion, labels_dictionary, class_selection, no_overlap, OFFSET):

    temp_session_data = np.zeros((1, channels, int(window_size)))
    temp_session_labels = np.zeros((1, 1))

    for n in range(1, max_sessions + 1):
        Session_number = n

        for x in range(1, max_blocks + 1):
            Block_number = x

            EEG = mne.io.read_raw_eeglab(base_file_path + subject_name + processing_selection + '/Session_' + str(
                Session_number) + '/Block_' + str(Block_number) + '/' + identifier + 'Block_' + str(
                Block_number) + '.set')
            EEG_data = EEG.get_data()

            if filter_data == 1:
                EEG_data = mne.filter.filter_data(EEG_data, sfreq, l_freq, h_freq)

            # Get Events from one block
            filename = base_file_path + subject_name + '/Events/Events_phonemes_' + str(Session_number) + '_' + str(
                Block_number) + '.csv'
            block_events = np.genfromtxt(filename, dtype=str, delimiter=",", usecols=[0])
            label_indices = [i for i in range(len(block_events)) if block_events[i] != ""]

            # Get labels and data from a single block
            temp_block_labels, temp_block_data, count = get_block_labels(label_indices, channels, window_size,
                                                                         block_events, label_inclusion,
                                                                         labels_dictionary, class_selection, no_overlap,
                                                                         EEG_data, OFFSET)

            # Remove first entry for the block datas
            temp_block_data = temp_block_data[0:count]
            temp_block_labels = temp_block_labels[0:count]

            # Concetenate block data with the rest of the data
            temp_session_data = np.concatenate((temp_session_data, temp_block_data))
            temp_session_labels = np.concatenate((temp_session_labels, temp_block_labels))

    X = temp_session_data[1:len(temp_session_data)]

    # pdb.set_trace()
    Y = np.squeeze(temp_session_labels[1:len(temp_session_labels)])
    return X, Y