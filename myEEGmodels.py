import pdb
import tensorflow
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Activation, Permute, Dropout, Bidirectional, LSTM, GRU
from keras.layers import Conv1D, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import SeparableConv2D, DepthwiseConv2D, MultiHeadAttention
from keras.layers import BatchNormalization, LayerNormalization, GlobalAveragePooling1D
from keras.layers import Input, Flatten
from keras.constraints import max_norm

def main_CNN_general(nb_classes=3, Chans=64, Samples=256,
                 dropoutRate=0.5, max_conv_blocks=2, con_type='dense', num_mlp_units=[256], hidden_units=32, sync_on=0, output_type='discrete', final_filter_size = 50, first_filter=10):

    if max_conv_blocks == 1:
        input_main = Input((Chans, Samples, 1))

        block_start = Conv2D(final_filter_size, (1, first_filter),
                             input_shape=(Chans, Samples, 1),
                             kernel_constraint=max_norm(2., axis=(0, 1, 2)), padding='same')(input_main)
        block_start = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=1,
                                      depthwise_constraint=max_norm(1.))(block_start)

        block_start = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block_start)
        block_start = Activation('relu')(block_start)
        block_start = AveragePooling2D(pool_size=(1, 4), strides=(1, 4))(block_start)

        blockn=block_start

    else:
        input_main = Input((Chans, Samples, 1))

        block_start = Conv2D(hidden_units, (1, first_filter),
                             input_shape=(1, Chans, Samples),
                             kernel_constraint=max_norm(2., axis=(0, 1, 2)), padding='same')(input_main)

        block_start = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=1,
                                      depthwise_constraint=max_norm(1.))(block_start)

        block_start = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block_start)
        block_start = Activation('relu')(block_start)
        block_start = AveragePooling2D(pool_size=(1, 4), strides=(1, 4))(block_start)

        for x in range(0,max_conv_blocks-1):
            blockn = Dropout(dropoutRate)(block_start)

            if x == max_conv_blocks-2:
                blockn = Conv2D(final_filter_size, (1, int(first_filter/4)),kernel_constraint=max_norm(2., axis=(0, 1, 2)), padding='same')(blockn)
            else:
                blockn = Conv2D(hidden_units, (1, int(first_filter/4)), kernel_constraint=max_norm(2., axis=(0, 1, 2)),padding='same')(blockn)
            blockn = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(blockn)
            blockn = Activation('relu')(blockn)


            if x != max_conv_blocks-2:
                if con_type == 'dense':
                    blockn = keras.layers.Concatenate()([block_start, blockn])
                elif con_type == 'residual':
                    blockn = keras.layers.Add()([block_start, blockn])

            block_start = blockn

    blockn = Flatten()(blockn)
    for dim in num_mlp_units:
        blockn = Dense(dim)(blockn)

    if output_type == 'discrete':
        outputs = Dense(nb_classes, activation="softmax")(blockn)
    elif output_type == 'cont':
        outputs = Dense(nb_classes, activation="linear")(blockn)


    return Model(inputs=input_main, outputs=outputs)

def main_RNN_general(nb_classes=3, Chans=64, Samples=256,
                 dropout_rate=0.5, hidden_units=100, num_rnn_layers=4, num_mlp_units = [1024],
                                 RNN_type = 'LSTM', bidir = 1, con_type = 'dense', output_type='discrete', final_layer=234):

    if num_rnn_layers ==1:
        # start the model
        input_main = Input((Samples, Chans))
        block_start = input_main

        if bidir == 1:
            if RNN_type == 'LSTM':
                block_start = Bidirectional(LSTM(int(final_layer/2)))(block_start)
            else:
                block_start = Bidirectional(GRU(int(final_layer/2)))(block_start)
        else:

            if RNN_type == 'LSTM':
                block_start = (LSTM(final_layer))(block_start)
            else:
                block_start = (GRU(final_layer)(block_start))

        block_start = LayerNormalization(epsilon=1e-05)(block_start)
        block_start = Activation('relu')(block_start)
        blockn = Dropout(dropout_rate)(block_start)

    else:

        # start the model
        input_main = Input((Samples, Chans))
        block_start = input_main

        if bidir == 1:
            if RNN_type == 'LSTM':
                block_start = Bidirectional(LSTM(hidden_units, return_sequences=True))(block_start)
            else:
                block_start = Bidirectional(GRU(hidden_units, return_sequences=True))(block_start)
        else:

            if RNN_type == 'LSTM':
                block_start = (LSTM(hidden_units, return_sequences=True))(block_start)
            else:
                block_start = (GRU(hidden_units, return_sequences=True)(block_start))


        block_start = LayerNormalization(epsilon=1e-05)(block_start)
        block_start = Activation('relu')(block_start)
        block_start = Dropout(dropout_rate)(block_start)

        for xx in range(0, num_rnn_layers-1):

            if xx == num_rnn_layers-2:
                if bidir == 1:
                    if RNN_type == 'LSTM':
                        blockn = Bidirectional(LSTM(int(final_layer / 2)))(block_start)
                    else:
                        blockn = Bidirectional(GRU(int(final_layer / 2)))(block_start)
                else:

                    if RNN_type == 'LSTM':
                        blockn = (LSTM(final_layer))(block_start)
                    else:
                        blockn = (GRU(final_layer)(block_start))
            else:

                if bidir == 1:
                    if RNN_type == 'LSTM':
                        blockn = Bidirectional(LSTM(hidden_units, return_sequences=True))(block_start)
                    else:
                        blockn = Bidirectional(GRU(hidden_units, return_sequences=True))(block_start)
                else:

                    if RNN_type == 'LSTM':
                        blockn = (LSTM(hidden_units, return_sequences=True))(block_start)
                    else:
                        blockn = (GRU(hidden_units, return_sequences=True))(block_start)

            blockn = LayerNormalization()(blockn)
            blockn = Activation('relu')(blockn)
            blockn = Dropout(dropout_rate)(blockn)


            if xx != num_rnn_layers-2:
                if con_type == 'dense':
                    blockn = keras.layers.Concatenate()([block_start, blockn])
                elif con_type == 'residual':
                    blockn = keras.layers.Add()([block_start, blockn])

            block_start = blockn

    for dim in num_mlp_units:
        blockn = Dense(dim)(blockn)


    if output_type == 'discrete':
        final = Dense(nb_classes, activation="softmax")(blockn)
    elif output_type == 'cont':
        final = Dense(nb_classes, activation="linear")(blockn)


    return Model(inputs=input_main, outputs=final)

def transformer_RNN_EEG_general(head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0,
                             Chans=64, Samples=60, n_classes=5,
                             output_type='discrete', opt_model=1):

    if opt_model == 1:
        RNN_type = 'GRU'
        RNN_dropout_rate = 0.5
        num_rnn_layers = 2
        con_type = 'residual'
        bidir = 0
        hidden_units = 64
        final_layer = 720
    else:
        RNN_type = 'LSTM'
        RNN_dropout_rate = 0.5
        num_rnn_layers = 1
        con_type = 'residual'
        bidir = 0
        hidden_units = 128
        final_layer = 720

    if num_rnn_layers ==1:
        # start the model
        input_main = Input((Samples, Chans))
        block_start = input_main

        if bidir == 1:
            if RNN_type == 'LSTM':
                block_start = Bidirectional(LSTM(int(final_layer/2)))(block_start)
            else:
                block_start = Bidirectional(GRU(int(final_layer/2)))(block_start)
        else:

            if RNN_type == 'LSTM':
                block_start = (LSTM(final_layer))(block_start)
            else:
                block_start = (GRU(final_layer)(block_start))

        block_start = LayerNormalization(epsilon=1e-05)(block_start)
        block_start = Activation('relu')(block_start)
        blockn = Dropout(RNN_dropout_rate)(block_start)

    else:

        # start the model
        input_main = Input((Samples, Chans))
        block_start = input_main


        if bidir == 1:
            if RNN_type == 'LSTM':
                block_start = Bidirectional(LSTM(hidden_units, return_sequences=True))(block_start)
            else:
                block_start = Bidirectional(GRU(hidden_units, return_sequences=True))(block_start)
        else:

            if RNN_type == 'LSTM':
                block_start = (LSTM(hidden_units, return_sequences=True))(block_start)
            else:
                block_start = (GRU(hidden_units, return_sequences=True)(block_start))


        block_start = LayerNormalization(epsilon=1e-05)(block_start)
        block_start = Activation('relu')(block_start)
        block_start = Dropout(RNN_dropout_rate)(block_start)

        for xx in range(0, num_rnn_layers-1):

            if xx == num_rnn_layers-2:
                if bidir == 1:
                    if RNN_type == 'LSTM':
                        blockn = Bidirectional(LSTM(int(final_layer / 2)))(block_start)
                    else:
                        blockn = Bidirectional(GRU(int(final_layer / 2)))(block_start)
                else:

                    if RNN_type == 'LSTM':
                        blockn = (LSTM(final_layer))(block_start)
                    else:
                        blockn = (GRU(final_layer))(block_start)
            else:

                if bidir == 1:
                    if RNN_type == 'LSTM':
                        blockn = Bidirectional(LSTM(hidden_units, return_sequences=True))(block_start)
                    else:
                        blockn = Bidirectional(GRU(hidden_units, return_sequences=True))(block_start)
                else:

                    if RNN_type == 'LSTM':
                        blockn = (LSTM(hidden_units, return_sequences=True))(block_start)
                    else:
                        blockn = (GRU(hidden_units, return_sequences=True))(block_start)

            blockn = LayerNormalization()(blockn)
            blockn = Activation('relu')(blockn)
            blockn = Dropout(RNN_dropout_rate)(blockn)


            if xx != num_rnn_layers-2:
                if con_type == 'dense':
                    blockn = keras.layers.Concatenate()([block_start, blockn])
                elif con_type == 'residual':
                    blockn = keras.layers.Add()([block_start, blockn])

            block_start = blockn

    x = blockn
    x = tensorflow.keras.backend.expand_dims(x,axis=-1)

    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)

    for dim in mlp_units:
        x = Dense(dim)(x)
        x = Dropout(dropout)(x)

    if output_type == 'discrete':
        final = Dense(n_classes, activation="softmax")(x)
    elif output_type == 'cont':
        final = Dense(n_classes, activation="linear")(x)

    return Model(input_main, final)

def transformer_CNN_EEG_general( head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0,
                            Chans=64, Samples=60, n_classes=5,
                             output_type='discrete', opt_model=1):

    if opt_model == 1:
        first_filter = 10
        CNN_dropout_rate = 0.5
        max_conv_blocks = 2
        con_type = 'residual'
        hidden_units = 64
        final_filter_size = 48
    else:
        first_filter = 5
        CNN_dropout_rate = 0.5
        max_conv_blocks = 1
        con_type = 'residual'
        hidden_units = 32
        final_filter_size = 48

    if max_conv_blocks == 1:
        input_main = Input((Chans, Samples, 1))

        block_start = Conv2D(final_filter_size, (1, first_filter),
                             input_shape=(Chans, Samples, 1),
                             kernel_constraint=max_norm(2., axis=(0, 1, 2)), padding='same')(input_main)
        block_start = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=1,
                                      depthwise_constraint=max_norm(1.))(block_start)

        block_start = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block_start)
        block_start = Activation('relu')(block_start)
        block_start = AveragePooling2D(pool_size=(1, 4), strides=(1, 4))(block_start)

        blockn=block_start

    else:
        input_main = Input((Chans, Samples, 1))

        block_start = Conv2D(hidden_units, (1, first_filter),
                             input_shape=(1, Chans, Samples),
                             kernel_constraint=max_norm(2., axis=(0, 1, 2)), padding='same')(input_main)
        block_start = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=1,
                                      depthwise_constraint=max_norm(1.))(block_start)


        block_start = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block_start)
        block_start = Activation('relu')(block_start)
        block_start = AveragePooling2D(pool_size=(1, 4), strides=(1, 4))(block_start)

        for x in range(0,max_conv_blocks-1):
            blockn = Dropout(CNN_dropout_rate)(block_start)

            if x == max_conv_blocks-2:
                blockn = Conv2D(final_filter_size, (1, int(first_filter/4)),kernel_constraint=max_norm(2., axis=(0, 1, 2)), padding='same')(blockn)
            else:
                blockn = Conv2D(hidden_units, (1, int(first_filter/4)), kernel_constraint=max_norm(2., axis=(0, 1, 2)),padding='same')(blockn)
            blockn = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(blockn)
            blockn = Activation('relu')(blockn)


            if x != max_conv_blocks-2:
                if con_type == 'dense':
                    blockn = keras.layers.Concatenate()([block_start, blockn])
                elif con_type == 'residual':
                    blockn = keras.layers.Add()([block_start, blockn])

            block_start = blockn

    x = Flatten()(blockn)

    x = tensorflow.keras.backend.expand_dims(x, axis=-1)


    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format="channels_first")(x)

    for dim in mlp_units:
        x = Dense(dim)(x)
        x = Dropout(dropout)(x)

    if output_type == 'discrete':
        final = Dense(n_classes, activation="softmax")(x)
    elif output_type == 'cont':
        final = Dense(n_classes, activation="linear")(x)

    return Model(input_main, final)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):

    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

