from myEEGmodels import *
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pdb
import mne
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import ParameterGrid
from keras import backend as K
from my_utils import *
from keras.optimizers import adam_v2
import gc
import sklearn
import sklearn.dummy
import time
from time import gmtime, strftime, localtime

subject_names = ["Sbj1", "Sbj2", "Sbj3", "Sbj4", "Sbj5", "Sbj6", "Sbj7", "Sbj8", "Sbj9", "Sbj10"]
max_sessions_array = [2, 1, 1, 1, 1, 1, 1, 1, 2, 1]
max_blocks_array = [4, 4, 6, 4, 6, 4, 4, 4, 4, 3]

subject_select = 0
selected_subjects = np.arange(0:len(subject_names))

if subject_select == 1:
    subject_range = selected_subjects
else:
    subject_range = range(0, len(subject_names))

plotting = 0

training_model = 1

for kk in subject_range:

    subject_name = subject_names[kk]
    max_sessions = max_sessions_array[kk]
    max_blocks = max_blocks_array[kk]

    # Cleaning Types
    cleaning_types = ['EEMD_CCA', 'EEMD_ICA', 'EEMD_alone', 'CCA', 'ICA', 'no_clean']

    select_specific = 0
    specific_type = 'EEMD_95_85_'

    # Performance Labels
    performance_labels = ['Subject Name', 'Max Session', 'Max Blocks', 'Identifier', 'Train MSE', 'Validation MSE', 'Test MSE', 'Test MCD', 'Dummy MCD', 'Window Size', 'Low Freq', 'High Freq', 'Last Epoch', 'Param_grid', 'time_trained']
    performance_data = pd.DataFrame([], columns=performance_labels)

    # Input data size formatting
    window_size = 60 # in time points of 200Hz
    channels = 61
    do_CSP = 0
    csp_total_rem_chan = 30

    # Filtering configuration
    filter_data = 0
    sfreq = 200
    l_freq = .1
    h_freq = 100

    # General deep learning parameters
    model_types = ['CNN','RNN','TRAN_CNN','TRAN_RNN']
    batch_size = 64
    learning_rate = 0.00005
    earlystopping_patience = 5
    learning_rate_BASE = 0.0001
    learning_rate_TRAN = 0.0001
    epoch_count = 20
    verbosity = 1

    # MFCC specific parameters
    include_some_rest = 0
    collected_MFCC_count = 25
    MFCC_count = 25

    # Testing parameters
    do_flip_identifiers = 0
    val_split_shuffle = 0
    testing_EMG = 1
    select_group_identifiers = 1


    performances_save_folder = "C:/Users/Speech/Code_rev2/Python_Code/Continuous_predictions/"
    base_file_path = "D:/Speech_Collection/Subject_data/"

    checkpointPath = performances_save_folder + "Saved_models/check" + strftime("-%Y-%m-%d-%H-%M-%S",
                                                                                localtime()) + str(kk) + ".hdf5"
    save_performances = 1
    save_MFCCs = 1
    save_file = performances_save_folder + subject_name + "/Performances/Continuous_predictions" + strftime("-%Y-%m-%d-%H-%M-%S", localtime()) + '.csv'

    if testing_EMG == 1:
        test_option = "_final5"
    else:
        test_option = ""

    identifiers, processing_selections = get_identifiers(cleaning_types, base_file_path, subject_name, test_option)
    if do_flip_identifiers == 1:
        identifiers = np.flip(identifiers)
        processing_selections = np.flip(processing_selections)

    group_specification = [np.where(identifiers == 'EEMD_corr_CCA_95_79_')[0][0],
                           np.where(identifiers == 'EEMD_PSD_ICA_95_9_')[0][0],
                           np.where(identifiers == 'ICA_iclabel_86_')[0][0], np.where(identifiers == '')[0][0]]

    if select_group_identifiers == 1:
        processing_selections = processing_selections[group_specification]
        identifiers = identifiers[group_specification]

    if select_specific == 1:
        identifiers = [specific_type]

    for c in range(0, len(identifiers)):

        identifier = identifiers[c]
        processing_selection = processing_selections[c]

        temp_session_data = np.zeros((1, channels, int(window_size)))
        temp_session_labels = np.zeros((1, collected_MFCC_count))

        for n in range(1, max_sessions+1):
            Session_number = n
            for x in range(1, max_blocks+1):

                Block_number = x

                # Get EEG from one block
                EEG = mne.io.read_raw_eeglab(
                    base_file_path + subject_name + processing_selection + '/Session_' + str(Session_number) + '/Block_' + str(
                        Block_number) + '/' + identifier + 'Block_' + str(Block_number) + '.set')
                EEG_data = EEG.get_data()

                if filter_data == 1:
                    EEG_data = mne.filter.filter_data(EEG_data, sfreq, l_freq, h_freq)

                # Get MFCC from one block
                MFCCs = np.transpose(np.genfromtxt(base_file_path + subject_name + '/MFCC/Session_' + str(Session_number) + '/Block_' + str(Block_number) + '/MFCC' + str(Session_number) + '_' + str(Block_number) + '.csv', delimiter=','))

                # Get EEG Events
                filename = base_file_path + subject_name + '/Events/Events_speech_' + str(Session_number) + '_' + str(Block_number) + '.csv'
                EEG_events = np.loadtxt(filename, delimiter=",")

                # Initialize block data and labels arrays
                temp_block_data = np.zeros((1, channels, int(window_size)))
                temp_block_labels = np.zeros((1, collected_MFCC_count))

                if include_some_rest==1:
                    add_rest = 600
                else:
                    add_rest = 0

                for i in range(len(EEG_events)):

                    if EEG_events[i,0] == 1:

                        temp_speech = np.zeros((int(EEG_events[i+1, 1] - EEG_events[i, 1] + add_rest), channels, int(window_size)))
                        temp_labels = np.zeros((int(EEG_events[i+1, 1] - EEG_events[i, 1] + add_rest), collected_MFCC_count))
                        count=0

                        for p in range(int(EEG_events[i, 1]), (int(EEG_events[i+1, 1]) + add_rest - 1)):

                            temp_EEG = np.array(EEG_data[:, int(p-(window_size/2)):int(p+(window_size/2))])
                            temp_label = np.array(MFCCs[:,p])
                            temp_speech[count] = temp_EEG
                            temp_labels[count] = temp_label
                            count += 1

                        # Concatenate trial data with Block data
                        temp_block_data = np.concatenate((temp_block_data, temp_speech))
                        temp_block_labels = np.concatenate((temp_block_labels, temp_labels))

                # Remove first entry for the block datas
                temp_block_data = temp_block_data[1:len(temp_block_data)-1]
                temp_block_labels = temp_block_labels[1:len(temp_block_labels)-1]

                temp_session_data = np.concatenate((temp_session_data, temp_block_data))
                temp_session_labels = np.concatenate((temp_session_labels, temp_block_labels))


        X = temp_session_data[1:len(temp_session_data)-1]
        Y = temp_session_labels[1:len(temp_session_labels)-1]

        Y = Y[:, 0:MFCC_count]

        Ch_names = range(0,61)
        print(Ch_names)
        X = X[:,Ch_names,:]

        test_split_percentage = 0.10
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_split_percentage, shuffle=False)
        del X
        if val_split_shuffle==1:
            val_split_percentage = .15
            X_train, X_val1, Y_train, Y_val1 = train_test_split(X_train, Y_train, test_size=val_split_percentage, shuffle=False)
            val_split_percentage = .10
            X_train, X_val2, Y_train, Y_val2 = train_test_split(X_train, Y_train, test_size=val_split_percentage, shuffle=True)
            X_val = np.concatenate((X_val1, X_val2), axis=0)
            Y_val = np.concatenate((Y_val1, Y_val2), axis=0)
        else:
            val_split_percentage = .25
            X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=val_split_percentage, shuffle=False)

        scalers = {}
        for i in range(X_train.shape[2]):
            scalers[i] = StandardScaler()
            X_train[:, :, i] = scalers[i].fit_transform(X_train[:, :, i])

        for i in range(X_test.shape[2]):
            X_test[:, :, i] = scalers[i].transform(X_test[:, :, i])

        for i in range(X_val.shape[2]):
            X_val[:, :, i] = scalers[i].transform(X_val[:, :, i])

        scalers_output = {}
        for i in range(Y_train.shape[1]):
            scalers_output[i] = StandardScaler()
            Y_train[:, i] = np.squeeze(scalers_output[i].fit_transform(Y_train[:, i].reshape(-1, 1)))

        for i in range(Y_test.shape[1]):
            Y_test[:, i] = np.squeeze(scalers_output[i].transform(Y_test[:, i].reshape(-1, 1)))

        for i in range(Y_val.shape[1]):
            Y_val[:, i] = np.squeeze(scalers_output[i].transform(Y_val[:, i].reshape(-1, 1)))

        model_list = []
        param_list = []

        for p in range(len(model_types)):

            model_type = model_types[p]

            if model_type == 'RNN':

                hidden_units_list = [64]
                dropout_rate_list = [0.5]
                num_rnn_layers_list = [2]
                mlp_units_list = [[64]]  # first run had both 64 and 128/64
                mult_direction_list = ['ones']
                RNN_type_list = ['GRU']
                con_type_list = ['residual']
                bidir_list = [0]
                output_type_list = ['cont']
                final_layer_list = [720]

                param_grid = ParameterGrid(
                    {'1_model_type': [model_type], 'dropoutRate': dropout_rate_list, 'hidden_units': hidden_units_list,
                     'num_rnn_layers': num_rnn_layers_list,
                     'num_mlp_units': mlp_units_list, 'mult_direction': mult_direction_list, 'RNN_type': RNN_type_list,
                     'bidir': bidir_list, 'con_type': con_type_list, 'output_type': output_type_list,
                     'final_layer': final_layer_list})

                for i in range(len(param_grid)):
                    param_temp = param_grid[i]
                    param_list.append(param_temp)

            elif model_type == 'CNN':

                dropoutRate_list = [0.5]  # first run with heather's data was 0.25 and 0.5, but didn't seem to matter and needed to reduce the parameter variations
                max_conv_blocks_list = [2]
                con_type_list = ['residual']
                mlp_units_list = [[64]]
                hidden_units_list = [64]
                sync_on_list = [0]
                output_type = ['cont']
                final_filter_size_list = [48]
                first_filter_list = [10]

                param_grid = ParameterGrid(
                    {'1_model_type': [model_type], 'dropoutRate': dropoutRate_list, 'max_conv_blocks': max_conv_blocks_list,
                     'con_type': con_type_list, 'num_mlp_units': mlp_units_list, 'hidden_units': hidden_units_list,
                     'sync_on': sync_on_list, 'output_type': output_type, 'final_filter_size': final_filter_size_list,
                     'first_filter': first_filter_list})

                for i in range(len(param_grid)):
                    param_temp = param_grid[i]
                    param_list.append(param_temp)

            elif model_type == 'TRAN_RNN':


                head_size_list = [240]  # default was 256
                num_heads_list = [3]  # default was 4
                ff_dim_list = [5]  # default was 4
                num_transformer_blocks_list = [4]  # default was 4
                mlp_units_list = [[64]]  # default was 1024 (more items in a list means more layers
                dropout_list = [0.5]  # default was .25
                opt_model_list = [1]

                output_type = ['cont']  # discrete or cont

                param_grid = ParameterGrid(
                    {'1_model_type': [model_type], 'head_size': head_size_list, 'num_heads': num_heads_list,
                     'ff_dim': ff_dim_list, 'num_transformer_blocks': num_transformer_blocks_list,
                     'mlp_units': mlp_units_list,
                     'dropout': dropout_list,
                     'output_type': output_type, 'opt_model': opt_model_list})

                for i in range(len(param_grid)):
                    param_temp = param_grid[i]
                    param_list.append(param_temp)

            elif model_type == 'TRAN_CNN':



                head_size_list = [240]  # default was 256
                num_heads_list = [3]  # default was 4
                ff_dim_list = [5]  # default was 4
                num_transformer_blocks_list = [2]  # default was 4
                mlp_units_list = [[64]]  # default was 1024 (more items in a list means more layers
                dropout_list = [0.5]  # default was .25
                opt_model_list = [1]

                output_type = ['cont']  # discrete or cont

                param_grid = ParameterGrid(
                    {'1_model_type': [model_type], 'head_size': head_size_list, 'num_heads': num_heads_list,
                     'ff_dim': ff_dim_list, 'num_transformer_blocks': num_transformer_blocks_list,
                     'mlp_units': mlp_units_list,
                     'dropout': dropout_list,
                     'output_type': output_type, 'opt_model': opt_model_list})

                for i in range(len(param_grid)):
                    param_temp = param_grid[i]
                    param_list.append(param_temp)

        for i in range(len(param_list)):

            parameters = param_list[i]
            print(parameters)
            print('Model # ' + str(i) + ' of ' + str(len(param_list)), ' total models.')
            print(identifier)
            print(subject_name)


            if training_model == 1:
                # Create model from list of parameters
                if parameters['1_model_type'] == 'CNN':
                    if (len(X_test.shape) == 3):
                        if X_test.shape[1] == window_size:
                            X_train = np.swapaxes(X_train, 1, 2)
                            X_test = np.swapaxes(X_test, 1, 2)
                            X_val = np.swapaxes(X_val, 1, 2)

                        X_train = np.expand_dims(X_train, 3)
                        X_test = np.expand_dims(X_test, 3)
                        X_val = np.expand_dims(X_val, 3)

                    learning_rate = learning_rate_BASE

                    model = main_CNN_general(nb_classes=MFCC_count, Chans=len(Ch_names), Samples=window_size,
                                             dropoutRate=parameters['dropoutRate'],
                                             max_conv_blocks=parameters['max_conv_blocks'], con_type=parameters['con_type'],
                                             num_mlp_units=parameters['num_mlp_units'], hidden_units=parameters['hidden_units'],
                                             sync_on=parameters['sync_on'], output_type=parameters['output_type'],
                                             final_filter_size=parameters['final_filter_size'],
                                             first_filter=parameters['first_filter'])


                elif parameters['1_model_type'] == 'RNN':

                    if len(X_test.shape) >= 4:
                        X_train = np.squeeze(X_train)
                        X_test = np.squeeze(X_test)
                        X_val = np.squeeze(X_val)

                        if X_test.shape[1] == channels:
                            X_train = np.swapaxes(X_train, 1, 2)
                            X_test = np.swapaxes(X_test, 1, 2)
                            X_val = np.swapaxes(X_val, 1, 2)
                    elif X_test.shape[1] == channels and len(X_test.shape) == 3:
                        X_train = np.swapaxes(X_train, 1, 2)
                        X_test = np.swapaxes(X_test, 1, 2)
                        X_val = np.swapaxes(X_val, 1, 2)

                    learning_rate = learning_rate_BASE
                    model = main_RNN_general(nb_classes=MFCC_count, Chans=len(Ch_names), Samples=window_size,
                                             dropout_rate=parameters['dropoutRate'], hidden_units=parameters['hidden_units'],
                                             num_rnn_layers=parameters['num_rnn_layers'],
                                             num_mlp_units=parameters['num_mlp_units'],
                                             RNN_type=parameters['RNN_type'], bidir=parameters['bidir'],
                                             con_type=parameters['con_type'], output_type=parameters['output_type'],
                                             final_layer=parameters['final_layer'])

                elif parameters['1_model_type'] == 'TRAN_RNN':

                    if len(X_test.shape) >= 4:
                        X_train = np.squeeze(X_train)
                        X_test = np.squeeze(X_test)
                        X_val = np.squeeze(X_val)

                        if X_test.shape[1] == channels:
                            X_train = np.swapaxes(X_train, 1, 2)
                            X_test = np.swapaxes(X_test, 1, 2)
                            X_val = np.swapaxes(X_val, 1, 2)

                    elif X_test.shape[1] == channels and len(X_test.shape) == 3:
                        X_train = np.swapaxes(X_train, 1, 2)
                        X_test = np.swapaxes(X_test, 1, 2)
                        X_val = np.swapaxes(X_val, 1, 2)

                    learning_rate = learning_rate_TRAN

                    model = transformer_RNN_EEG_general(
                        head_size=parameters['head_size'],
                        num_heads=parameters['num_heads'],
                        ff_dim=parameters['ff_dim'],
                        num_transformer_blocks=parameters['num_transformer_blocks'],
                        mlp_units=parameters['mlp_units'],
                        dropout=parameters['dropout'],
                        Chans=len(Ch_names),
                        Samples=window_size,
                        n_classes=MFCC_count,
                        output_type=parameters['output_type'],
                        opt_model=parameters['opt_model'])

                elif parameters['1_model_type'] == 'TRAN_CNN':

                    if (len(X_test.shape) == 3):
                        if X_test.shape[1] == window_size:
                            X_train = np.swapaxes(X_train, 1, 2)
                            X_test = np.swapaxes(X_test, 1, 2)
                            X_val = np.swapaxes(X_val, 1, 2)

                        X_train = np.expand_dims(X_train, 3)
                        X_test = np.expand_dims(X_test, 3)
                        X_val = np.expand_dims(X_val, 3)

                    learning_rate = learning_rate_TRAN

                    model = transformer_CNN_EEG_general(
                        head_size=parameters['head_size'],
                        num_heads=parameters['num_heads'],
                        ff_dim=parameters['ff_dim'],
                        num_transformer_blocks=parameters['num_transformer_blocks'],
                        mlp_units=parameters['mlp_units'],
                        dropout=parameters['dropout'],
                        Chans=len(Ch_names),
                        Samples=window_size,
                        n_classes=MFCC_count,
                        output_type=parameters['output_type'],
                        opt_model=parameters['opt_model'])

                opt = adam_v2.Adam(learning_rate=learning_rate)
                model.compile(loss='mse', optimizer=opt, metrics=[tf.keras.metrics.MeanSquaredError()])

                # can get the number of parameters in the model
                numParams = model.count_params()


                # print out a text summary of the architecture
                model.summary()

                # define the ModelCheckpoint and earlystopping callback
                checkpointer = ModelCheckpoint(filepath=checkpointPath, verbose=verbosity, save_best_only=True, monitor='val_loss')
                earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=earlystopping_patience)

                # fit model
                fittedModel = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epoch_count,
                                        verbose=verbosity, validation_data=(X_val, Y_val),
                                        callbacks=[checkpointer, earlystopping], shuffle='yes')

                # Load the model with the best validation loss, and evaluate
                model.load_weights(checkpointPath)

                # Get scores with loaded model
                score = model.evaluate(X_test, Y_test, verbose=0)
                score_train = model.evaluate(X_train, Y_train, verbose=0)
                score_val = model.evaluate(X_val, Y_val, verbose=0)

                probs = model.predict(X_test)
                probs_train = model.predict(X_train)
                probs_val = model.predict(X_val)

                print('Train MSE:', score_train[1], ' with ', len(np.unique(probs_train)), ' unique values (original unique values: ', len(np.unique(Y_train)), ')')
                print('Validation MSE:', score_val[1], ' with ', len(np.unique(probs_val)), ' unique values (original unique values: ', len(np.unique(Y_val)), ')')
                print('Test MSE:', score[1], ' with ', len(np.unique(probs)), ' unique values (original unique values: ', len(np.unique(Y_test)), ')')


                for i in range(probs.shape[1]):
                    probs[:, i] = np.squeeze(scalers_output[i].inverse_transform(probs[:, i].reshape(-1, 1)))
                    Y_test[:, i] = np.squeeze(scalers_output[i].inverse_transform(Y_test[:, i].reshape(-1, 1)))


                MCD_test = 10 * np.sqrt(2) / np.log(10) * np.mean(np.sqrt(np.sum(np.square(Y_test - probs), 1)))
                print('Test MFCC:', MCD_test)

                for i in range(Y_test.shape[1]):
                    Y_test[:, i] = np.squeeze(scalers_output[i].transform(Y_test[:, i].reshape(-1, 1)))
                    probs[:, i] = np.squeeze(scalers_output[i].transform(probs[:, i].reshape(-1, 1)))

                if plotting==1:
                # Plot training & validation loss values
                    plt.plot(fittedModel.history['loss'], label = 'Training Loss')
                    plt.plot(fittedModel.history['val_loss'], label = 'Validation Loss')
                    plt.title('Model loss')
                    plt.ylabel('Loss')
                    plt.xlabel('Epoch')
                    plt.legend(['Train', 'Validation'], loc='upper left')
                    plt.show()

                final_epoch = str(len(fittedModel.history['loss']))

                del model, fittedModel

            dummy = sklearn.dummy.DummyRegressor(strategy="mean")
            dummy.fit(X_train, Y_train)
            Y_pred_dummy = dummy.predict(X_test)

            for i in range(Y_pred_dummy.shape[1]):
                Y_pred_dummy[:, i] = np.squeeze(scalers_output[i].inverse_transform(Y_pred_dummy[:, i].reshape(-1, 1)))

            for i in range(Y_pred_dummy.shape[1]):
                Y_test[:, i] = np.squeeze(scalers_output[i].inverse_transform(Y_test[:, i].reshape(-1, 1)))


            MCD_dummy = 10 * np.sqrt(2) / np.log(10) * np.mean(np.sqrt(np.sum(np.square(Y_test - Y_pred_dummy), 1)))

            print('Dummy MFCC:', MCD_dummy, ' for subject', subject_name)
            current_time = strftime("-%Y-%m-%d-%H-%M-%S", localtime())

            if save_MFCCs == 1:

                for i in range(probs.shape[1]):
                    probs[:, i] = np.squeeze(scalers_output[i].inverse_transform(probs[:, i].reshape(-1, 1)))

                np.savetxt(performances_save_folder + subject_name + "/MFCCs/MFCC_preds" + current_time + ".csv", probs, delimiter=",")
                np.savetxt(performances_save_folder + subject_name + "/MFCCs/MFCC_true.csv", Y_test, delimiter=",")

            if save_performances==1:

                temp_performance = pd.DataFrame([(subject_name, str(max_sessions), str(max_blocks), identifier,
                                                  str(round(score_train[1], 4)), str(round(score_val[1], 4)),
                                                  str(round(score[1], 4)), str(MCD_test), str(MCD_dummy),
                                                  window_size, str(l_freq), str(h_freq),
                                                  final_epoch,
                                                  parameters, current_time)], columns=performance_labels)

                performance_data = pd.concat([performance_data, temp_performance])


                performance_data.to_csv(save_file)

            for i in range(Y_test.shape[1]):
                Y_test[:, i] = np.squeeze(scalers_output[i].transform(Y_test[:, i].reshape(-1, 1)))

            gc.collect()
            K.clear_session()

#pdb.set_trace()
