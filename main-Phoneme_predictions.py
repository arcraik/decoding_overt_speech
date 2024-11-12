import pdb
import sklearn.dummy
from myEEGmodels import *
import tensorflow
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from time import gmtime, strftime, localtime
from sklearn.model_selection import ParameterGrid
from keras import backend as K
from keras.utils.layer_utils import count_params
from my_utils import *
from keras.optimizers import adam_v2
import time
import gc
import scipy

# Subject info
subject_names = ["Sbj1", "Sbj2", "Sbj3", "Sbj4", "Sbj5", "Sbj6", "Sbj7", "Sbj8", "Sbj9", "Sbj10"]
max_sessions_array = [2, 1, 1, 1, 1, 1, 1, 1, 2, 1]
max_blocks_array = [4, 4, 6, 4, 6, 4, 4, 4, 4, 3]

select_subject = 0
selected_person = [0]

# Cleaning methods and options
cleaning_types = ['EEMD_CCA', 'EEMD_ICA', 'EEMD_alone', 'CCA', 'ICA', 'no_clean']
class_selections = ['moa', 'voice_noV', 'poa_v', 'poa_c', 'v_vs_c', '39_phones']

# Input data size formatting
window_size = 60 # in time points of 200Hz
channels = 61
do_CSP = 0
csp_total_rem_chan = 30

# Filtering configuration
filter_data = 0
sfreq = 200

if filter_data == 1:
    frequency_list = [[.1, 4], [4, 8], [8,12], [12,31], [31,70],[70, 100]]
    frequency_list = [ [4, 8], [8,12], [12,31], [31,70],[70, 100]]
else:
    frequency_list = [[0.1, 100]]

# General deep learning parameters
model_types = ['CNN']
batch_size = 64
learning_rate_BASE = 0.0005 #orig
learning_rate_TRAN = 0.00025
earlystopping_patience = 10
epoch_count = 50
verbosity = 1

# Testing Parameters
plotting = 1
dummy_plot_on = 0
save_cm = 1

OFFSET = 0
no_overlap = 0
test_random = 0

label_inclusion = ['_0', '_1']

# Set Base folders

performances_save_folder = "C:/Users//Speech/Code_rev2/Python_Code/"
base_file_path = "D:/Speech_Collection/Subject_data/"

# Set Performance Label Structure
saving_perf_data = 0

checkpointPath = performances_save_folder + "Saved_models/check" + strftime("-%Y-%m-%d-%H-%M-%S", localtime()) + ".hdf5"

if select_subject == 1:
    subject_range = selected_person
else:
    subject_range = range(0, len(subject_names))

for nn in range(0,len(frequency_list)):
    frequency_select = frequency_list[nn]
    l_freq = frequency_select[0]
    h_freq = frequency_select[1]

    for kk in subject_range:
        t = time.time()
        #pdb.set_trace()
        # Subject info
        subject_name = subject_names[kk]
        max_sessions = max_sessions_array[kk]
        max_blocks = max_blocks_array[kk]

        for ppp in range(0, len(class_selections)):

            performance_labels = ['Subject Name', 'Max Session', 'Max Blocks', 'Identifier', 'Train accuracy',
                                  'Validation accuracy', 'Test accuracy', 'CM Test', 'Dummy Model', 'CM Dummy',
                                  'Class Selection', 'Window Size',
                                  'Low Freq', 'High Freq', 'Last Epoch', 'model_param_count', 'full_model_summary',
                                  'Param_grid']
            performance_data = pd.DataFrame([], columns=performance_labels)

            class_selection = class_selections[ppp]
            save_file = performances_save_folder +"Discrete_performances/" + subject_name + "/" + class_selection + "/discrete_performance" + strftime("-%Y-%m-%d-%H-%M-%S", localtime()) + '.csv'

            testing_EMG = 1
            select_specific = 0
            specific_type = 'CCA_PSD_20_'
            select_group_identifiers = 1

            if testing_EMG == 1:
                test_option = "_final5"
            else:
                test_option = ""

            identifiers, processing_selections = get_identifiers(cleaning_types, base_file_path, subject_name, test_option)

            if select_specific == 1:
                identifiers = [specific_type]

            group_specification = [np.where(identifiers == 'EEMD_corr_CCA_95_79_')[0][0],
                                   np.where(identifiers == 'EEMD_PSD_ICA_95_9_')[0][0],
                                   np.where(identifiers == 'ICA_iclabel_86_')[0][0], np.where(identifiers == '')[0][0]]

            if select_group_identifiers == 1:
                processing_selections = processing_selections[group_specification]
                identifiers = identifiers[group_specification]

            labels_file = performances_save_folder + 'Labels_1_25_2023.csv'
            labels_dictionary = pd.read_csv(labels_file, skiprows = [1,2]).set_index('labels').T.to_dict()

            class_groups = pd.read_csv(labels_file, nrows=2).set_index('labels').T.to_dict()
            display_labels = class_groups['Classes'][class_selection].split(", ")
            display_title = class_groups['Category'][class_selection]

            for c in range(0, len(identifiers)):

                identifier = identifiers[c]
                processing_selection = processing_selections[c]

                X, Y = get_identifier_data(channels, window_size, max_sessions, max_blocks, base_file_path, subject_name, processing_selection, identifier, filter_data, sfreq, l_freq, h_freq, label_inclusion, labels_dictionary, class_selection, no_overlap, OFFSET)

                if do_CSP == 1:
                    CSP_check = mne.decoding.CSP(reg='ledoit_wolf', cov_est='epoch', n_components=8)
                    CSP_check.fit(X, Y)
                    csp_mean = np.mean(abs(CSP_check.filters_[:CSP_check.n_components]), axis=0)
                    csp_mean = np.argsort(csp_mean)
                    csp_best = np.where(csp_mean >= (X.shape[1]-csp_total_rem_chan))
                    Ch_names = csp_best[0].tolist()
                    del CSP_check
                else:
                    Ch_names = range(0, 61)

                X = X[:, Ch_names, :]
                test_split_percentage = 0.10
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_split_percentage, shuffle=False)
                print(X.shape)
                del X
                val_split_percentage = .25
                X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=val_split_percentage, shuffle=False)

                scalers = {}
                for i in range(X_train.shape[1]):
                    scalers[i] = StandardScaler()
                    X_train[:, i, :] = scalers[i].fit_transform(X_train[:, i, :])

                for i in range(X_test.shape[1]):
                    X_test[:, i, :] = scalers[i].transform(X_test[:, i, :])

                for i in range(X_val.shape[1]):
                    X_val[:, i, :] = scalers[i].transform(X_val[:, i, :])

                if test_random == 1:
                    np.random.shuffle(Y_train)

                class_weight = dict(enumerate(compute_class_weight(class_weight='balanced', classes=np.unique(Y_train), y=Y_train)))

                if verbosity != 0:
                    print(class_weight)
                    print(Ch_names)

                nb_classes = len(np.unique(Y))

                param_list = []

                for p in range(len(model_types)):

                    model_type = model_types[p]

                    if model_type == 'RNN':

                        hidden_units_list = [64, 96, 128]
                        dropout_rate_list = [0.5]
                        num_rnn_layers_list = [1, 2]
                        mlp_units_list = [[64]] #first run had both 64 and 128/64
                        mult_direction_list = ['ones']
                        RNN_type_list = ['GRU', 'LSTM']
                        con_type_list = ['none', 'residual']
                        bidir_list = [1, 0]
                        output_type_list = ['discrete']
                        final_layer_list = [240, 480, 720]

                        hidden_units_list = [64] #Optimal Model
                        dropout_rate_list = [0.5]
                        num_rnn_layers_list = [ 2]
                        mlp_units_list = [[64]] #first run had both 64 and 128/64
                        mult_direction_list = ['ones']
                        RNN_type_list = ['GRU']
                        con_type_list = ['residual']
                        bidir_list = [0]
                        output_type_list = ['discrete']
                        final_layer_list = [720]

                        param_grid = ParameterGrid(
                            {'1_model_type': [model_type], 'dropoutRate': dropout_rate_list, 'hidden_units': hidden_units_list, 'num_rnn_layers': num_rnn_layers_list,
                             'num_mlp_units': mlp_units_list, 'mult_direction': mult_direction_list, 'RNN_type': RNN_type_list,
                             'bidir': bidir_list, 'con_type': con_type_list, 'output_type': output_type_list, 'final_layer': final_layer_list})

                        for i in range(len(param_grid)):
                            param_temp = param_grid[i]
                            param_list.append(param_temp)

                    elif model_type == 'CNN':

                        dropoutRate_list = [0.5] #first run with heather's data was 0.25 and 0.5, but didn't seem to matter and needed to reduce the parameter variations
                        max_conv_blocks_list = [1, 2, 3, 4]
                        con_type_list = ['ones', 'residual']
                        mlp_units_list = [[64]]
                        hidden_units_list = [32, 48, 64]
                        sync_on_list = [0]
                        output_type = ['discrete']
                        final_filter_size_list = [16, 32, 48]
                        first_filter_list = [5, 10]

                        dropoutRate_list = [0.5] #Optimal model
                        max_conv_blocks_list = [2]
                        con_type_list = ['residual']
                        mlp_units_list = [[64]]
                        hidden_units_list = [64]
                        sync_on_list = [0]
                        output_type = ['discrete']
                        final_filter_size_list = [48]
                        first_filter_list = [10]

                        param_grid = ParameterGrid({'1_model_type': [model_type], 'dropoutRate': dropoutRate_list, 'max_conv_blocks': max_conv_blocks_list, 'con_type': con_type_list, 'num_mlp_units': mlp_units_list, 'hidden_units': hidden_units_list, 'sync_on': sync_on_list, 'output_type': output_type, 'final_filter_size':final_filter_size_list, 'first_filter':first_filter_list})

                        for i in range(len(param_grid)):
                            param_temp = param_grid[i]
                            param_list.append(param_temp)

                    elif model_type == 'TRAN_RNN':


                        head_size_list = [60, 120, 240]  # default was 256
                        num_heads_list = [4, 3, 2]  # default was 4
                        ff_dim_list = [5]  # default was 4
                        num_transformer_blocks_list = [4, 3, 2]  # default was 4
                        mlp_units_list = [[64], [128]]  # default was 1024 (more items in a list means more layers
                        dropout_list = [0.5]  # default was .25
                        opt_model_list = [1, 0]

                        head_size_list = [60, 120, 240]  # default was 256
                        num_heads_list = [4, 3, 2]  # default was 4
                        ff_dim_list = [5]  # default was 4
                        num_transformer_blocks_list = [4, 3, 2]  # default was 4
                        mlp_units_list = [[64]]  # default was 1024 (more items in a list means more layers
                        dropout_list = [0.5]  # default was .25
                        opt_model_list = [1, 0]

                        output_type = ['discrete']  # discrete or cont

                        param_grid = ParameterGrid(
                            {'1_model_type': [model_type], 'head_size': head_size_list, 'num_heads': num_heads_list,
                             'ff_dim': ff_dim_list, 'num_transformer_blocks': num_transformer_blocks_list,
                             'mlp_units': mlp_units_list,
                             'dropout': dropout_list,
                             'output_type': output_type, 'opt_model':opt_model_list})

                        for i in range(len(param_grid)):
                            param_temp = param_grid[i]
                            param_list.append(param_temp)

                    elif model_type == 'TRAN_CNN':

                        head_size_list = [60, 120, 240] # default was 256
                        num_heads_list = [4, 3, 2]  # default was 4
                        ff_dim_list = [5]  # default was 4
                        num_transformer_blocks_list = [4, 3, 2]  # default was 4
                        mlp_units_list = [[64], [128]]  # default was 1024 (more items in a list means more layers
                        dropout_list = [0.5]  # default was .25
                        opt_model_list = [1, 0]

                        head_size_list = [240]  # default was 256 OPTIMAL MODEL
                        num_heads_list = [3]  # default was 4
                        ff_dim_list = [5]  # default was 4
                        num_transformer_blocks_list = [2]  # default was 4
                        mlp_units_list = [[64]]  # default was 1024 (more items in a list means more layers
                        dropout_list = [0.5]  # default was .25
                        opt_model_list = [1]
                        output_type = ['discrete']  # discrete or cont

                        param_grid = ParameterGrid(
                            {'1_model_type': [model_type], 'head_size': head_size_list, 'num_heads': num_heads_list,
                             'ff_dim': ff_dim_list, 'num_transformer_blocks': num_transformer_blocks_list,
                             'mlp_units': mlp_units_list,
                             'dropout': dropout_list,
                              'output_type': output_type, 'opt_model':opt_model_list})

                        for i in range(len(param_grid)):
                            param_temp = param_grid[i]
                            param_list.append(param_temp)


                for i in range(len(param_list)):

                    parameters = param_list[i]
                    print(parameters)
                    print('Model # ' + str(i) + ' of ' + str(len(param_list)), ' total models.')
                    print(identifier)
                    print(subject_name)
                    print(class_selection)

                    # Create model from list of parameters
                    if parameters['1_model_type'] == 'CNN':
                        if (len(X_test.shape)==3):
                            if X_test.shape[1] == window_size:
                                X_train = np.swapaxes(X_train, 1, 2)
                                X_test = np.swapaxes(X_test, 1, 2)
                                X_val = np.swapaxes(X_val, 1, 2)

                            X_train = np.expand_dims(X_train, 3)
                            X_test = np.expand_dims(X_test, 3)
                            X_val = np.expand_dims(X_val, 3)

                        learning_rate = learning_rate_BASE

                        model = main_CNN_general(nb_classes=nb_classes, Chans=len(Ch_names), Samples=window_size,
                                 dropoutRate=parameters['dropoutRate'], max_conv_blocks=parameters['max_conv_blocks'], con_type= parameters['con_type'],
                                                  num_mlp_units=parameters['num_mlp_units'], hidden_units=parameters['hidden_units'], sync_on=parameters['sync_on'], output_type=parameters['output_type'], final_filter_size=parameters['final_filter_size'], first_filter=parameters['first_filter'])

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
                        model = main_RNN_general(nb_classes=nb_classes, Chans=len(Ch_names), Samples=window_size,
                                                  dropout_rate=parameters['dropoutRate'], hidden_units=parameters['hidden_units'],
                                                  num_rnn_layers=parameters['num_rnn_layers'],
                                                  num_mlp_units=parameters['num_mlp_units'],
                                                  RNN_type=parameters['RNN_type'], bidir=parameters['bidir'], con_type=parameters['con_type'], output_type=parameters['output_type'], final_layer = parameters['final_layer'])

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
                            n_classes=nb_classes,
                            output_type=parameters['output_type'],
                            opt_model=parameters['opt_model'])

                    elif parameters['1_model_type'] == 'TRAN_CNN':

                        if (len(X_test.shape)==3):
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
                            n_classes=nb_classes,
                            output_type=parameters['output_type'],
                            opt_model=parameters['opt_model'])

                    opt = adam_v2.Adam(learning_rate=learning_rate)
                    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=["accuracy"])


                    # can get the number of parameters in the model
                    numParams = model.count_params()

                    if verbosity != 0:
                        model.summary()

                    # define the ModelCheckpoint and earlystopping callback
                    checkpointer = ModelCheckpoint(filepath=checkpointPath, verbose=verbosity, save_best_only=True, monitor='val_loss')
                    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=earlystopping_patience)

                    # Fit model to data
                    fittedModel = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epoch_count,
                                            verbose=verbosity, validation_data=(X_val, Y_val),
                                            callbacks=[checkpointer, earlystopping], shuffle=True, class_weight=class_weight)

                    # Load the model with the best validation loss, and evaluate
                    model.load_weights(checkpointPath)

                    # Get scores with loaded model
                    score = model.evaluate(X_test, Y_test, verbose=0)
                    score_train = model.evaluate(X_train, Y_train, verbose=0)
                    score_val = model.evaluate(X_val, Y_val, verbose=0)

                    probs = model.predict(X_test)
                    probs_train = model.predict(X_train)
                    probs_val = model.predict(X_val)

                    pred = np.argmax(probs, axis=1)
                    pred_train = np.argmax(probs_train, axis=1)
                    pred_val = np.argmax(probs_val, axis=1)

                    print('Train accuracy:', score_train[1])
                    print('Validation accuracy:', score_val[1])
                    print('Test accuracy:', score[1])

                    if plotting == 1:
                        # Plot training & validation accuracy values
                        plt.plot(fittedModel.history['accuracy'], label='Training Accuracy')
                        plt.plot(fittedModel.history['val_accuracy'], label='Validation Accuracy')
                        plt.title('Model accuracy')
                        plt.ylabel('Accuracy')
                        plt.xlabel('Epoch')
                        plt.legend(['Train', 'Validation'], loc='upper left')
                        plt.show()

                        # Plot training & validation loss values
                        plt.plot(fittedModel.history['loss'], label='Training Loss')
                        plt.plot(fittedModel.history['val_loss'], label='Validation Loss')
                        plt.title('Model loss')
                        plt.ylabel('Loss')
                        plt.xlabel('Epoch')
                        plt.legend(['Train', 'Validation'], loc='upper left')
                        plt.show()

                    Y_test = np.array(Y_test)
                    Y_pred = np.array(pred)

                    cm = confusion_matrix(Y_test, Y_pred, normalize='true')

                    if save_cm == 1:
                        cm_notnoramlized = confusion_matrix(Y_test, Y_pred)
                        cm_filename = performances_save_folder + "Discrete_performances/CM/"+ subject_name + "_" + class_selection + "_" + identifier + str(int(score[1]*100)) + '.mat'
                        mdic = {'CM': cm_notnoramlized}
                        scipy.io.savemat(cm_filename, mdic)

                    if plotting == 1:
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
                        disp.plot(cmap=plt.cm.Blues)
                        disp.ax_.set_title(display_title)
                        fig = disp.ax_.get_figure()
                        fig.set_figwidth(20)
                        fig.set_figheight(20)
                        plt.show()

                    dummy_score_collection = []

                    for mmm in range(0, 5):
                        dummy = sklearn.dummy.DummyClassifier(strategy="stratified")
                        dummy.fit(X_train, Y_train)
                        Y_pred_dummy = dummy.predict(X_test)
                        score_dummy_temp = model.evaluate(X_test, Y_pred_dummy, verbose=0)

                        dummy_score_collection.append(score_dummy_temp[1])

                    score_dummy = np.mean(dummy_score_collection)

                    print('Dummy Test accuracy:', score_dummy)

                    cm2 = confusion_matrix(Y_test, Y_pred_dummy, normalize='all')
                    if dummy_plot_on == 1:
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm2,  display_labels=display_labels)
                        disp.plot(cmap=plt.cm.Blues)
                        disp.ax_.set_title("Chance model")
                        fig = disp.ax_.get_figure()
                        fig.set_figwidth(20)
                        fig.set_figheight(20)
                        plt.show()

                    #pdb.set_trace()

                    short_model_summary = []
                    model.summary(print_fn=lambda xx: short_model_summary.append(xx))
                    short_model_summary = "\n".join(short_model_summary)

                    temp_performance = pd.DataFrame([(subject_name, str(max_sessions), str(max_blocks), identifier, str(round(score_train[1], 4)), str(round(score_val[1], 4)), str(round(score[1], 4)), cm, str(round(score_dummy, 4)), cm2, class_selection, window_size, str(l_freq), str(h_freq), str(len(fittedModel.history['accuracy'])), str(count_params(model.trainable_weights)), short_model_summary, param_list[i])], columns=performance_labels)

                    performance_data = pd.concat([performance_data, temp_performance])

                    print('Saving Performance Data!')

                    if saving_perf_data == 1:
                        performance_data.to_csv(save_file)

                    del model, fittedModel
                    gc.collect()
                    # Deleting current model for speed's sake
                    K.clear_session()

        elapsed = time.time()-t
        print("time elapsed: " + str(elapsed))

    del X_train, X_val, X_test