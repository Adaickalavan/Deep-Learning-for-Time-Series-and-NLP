# Adaickalavan Meiyappan

import pdb
import numpy as np
import csv
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Lambda, Bidirectional, Conv1D, MaxPooling1D
import keras.callbacks

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def readIn(filepath):
    '''
    Arguments:
    filepath -- path of file to be read

    Returns:    
    x -- list of strings read from the input file    
    '''
    # Read in the files
    with open(filepath, 'r') as x:
        x = x.read()
        x = x.splitlines()  # Split files into sentences

    return x

def char2ix(x):
    '''
    Arguments:
    x -- list of strings
    
    Returns:
    char_to_ix -- dictionary mapping character to integer
    n_v -- vocabulary size
    '''
    # Create a python dictionary to map each character to an index 0-26.
    string = ''.join(x)
    chars = list(string)  # Create a list of unique characters (i.e., a to z and \n newline)
    chars_set = set(chars)
    n_v = len(chars_set)  # Vocabulary size
    char_to_ix = {ch: ii for ii, ch in enumerate(sorted(chars_set))}
    print(char_to_ix)

    return char_to_ix, n_v

def listOfStr_to_array(x, Tx, func):
    '''
    Arguments:
    x -- list of strings.  
    Tx -- maximum length of string
    func -- function to map characters to integers
    
    Returns:
    x_out -- numpy array. Shape = (m, Tx).
    '''

    x_out = np.ones((len(x), Tx), dtype=np.int64)*-1  # Create empty array to store one-hot encoding result
    for ii, string in enumerate(x):  # Iterate over each string in the file
        x_array = np.array(list(map(func, string)))  # Map characters to integers
        if len(x_array) <= Tx:
            x_out[ii, : len(x_array)] = x_array
        else:
            x_out[ii, :] = x_array[: Tx]

    return x_out

# Convert a list of characters into one-hot encoding numpy array
def convert_to_one_hot(x, n):
    '''
    Arguments:
    x -- array of integers. 
    n -- number of unique values. scalar value.
   
    Returns:
    y -- one hot encoded matrix of integers. shape = (len(x), n).
    '''

    y = np.eye(n)[x.reshape(-1)]
    return y

def one_hot(x, numOfVocab):
    '''
    Arguments:
    x -- input array of shape (m,Tx) where m = len(x)
    numOfVocab -- vocabulary size
    
    Returns:
    return -- output array of shape (m, Tx, numOfVocab)
    '''
    import tensorflow as tf
    return tf.to_float(tf.one_hot(x, numOfVocab, on_value=1, off_value=0, axis=-1))

def one_hot_outshape(x):
    '''
    Arguments:
    x -- input shape
    
    Returns:
    return -- one hot encoded shape of x, i.e., (m, Tx, numOfVocab)     
    '''
    return x[0], x[1], 26

def write2file(filename,resultList):
    '''
    Write the data 'resultList' in a single column format to the file named 'filename' 
    
    Arguments:
    filename -- name of file to write
    resultList -- data to be written            
    '''
    with open(filename,'w',newline='') as outputFile:
        writer = csv.writer(outputFile)
        for result in resultList:
            writer.writerow([result]) 

def main():

    # Read in the files
    x_traindev = readIn('./xtrain.txt')
    y_traindev = readIn('./ytrain.txt')

    # Throughout this python script, we shall use this notation:
    # m is number of examples
    # n_v is vocabulary size (i.e., vocab_size)
    # n_c is number of classes (i.e., number of novels to predict)
    # n_a is dimensionality of the hidden state in LSTM cell
    # Tx is length of each string. All strings will be appended or truncated to the same length

    # Compute the maximum string length
    Txmax = len(max(x_traindev, key=len))
    print(Txmax)
    # We set desired string length to 448 which is 5 char shorter than Txmax. 
    # This simplifies the dimensions during convolution later, without significant performance loss
    Tx = 448
    # Set the number of classes
    n_c = 12

    # Each sentence in the data set is a sequence of characters. Hence, we will 
    # build a character level sentence classification model.
    # First, build a character dictionary. Then, convert each training sample 
    # (i.e., sentence) from character sequence to integer sequence using the dictionary.
    char_to_ix, n_v = char2ix(x_traindev)
    # Convert x_traindev from character sequence to integer sequence
    x_traindev = listOfStr_to_array(x_traindev, Tx, lambda x: char_to_ix[x])
    # Check type and shape X_traindev
    print('X is of type {} and of shape {}'.format(type(x_traindev), x_traindev.shape))

    # Convert the y labels into integers
    y_traindev = np.array(list(map(int, y_traindev)))
    
    # Shuffle stratify-split the data into train and dev sets
    x_train, x_dev, y_train, y_dev = train_test_split(x_traindev, 
                                                      y_traindev,
                                                      test_size=0.20, 
                                                      random_state=1,
                                                      shuffle=True, 
                                                      stratify=y_traindev)
    # Check type and shape x_train, x_dev, y_train_onehot, y_dev_onehot
    print('x_train is of type {} and of shape {}'.format(type(x_train), x_train.shape))
    print('x_dev is of type {} and of shape {}'.format(type(x_dev), x_dev.shape))
    
    #Since the number of calsses are imbalanced, we compute and apply class weights to the cost function
    #Compute class weights using y_train     
    unique, counts = np.unique(y_train, return_counts=True)
    print('Class distribution:')
    print(dict(zip(unique, counts)))
    class_weight_vec = compute_class_weight('balanced',np.array(range(12)),y_train)
    class_weight_dict = {ii: w for ii, w in enumerate(class_weight_vec)}
    
    # Convert training and test labels to one hot matrices
    y_train_onehot = convert_to_one_hot(y_train, n_c)
    y_dev_onehot = convert_to_one_hot(y_dev, n_c)
    print('y_train_onehot is of type {} and of shape {}'.format(type(y_train_onehot), y_train_onehot.shape))
    print('y_dev_onehot is of type {} and of shape {}'.format(type(y_dev_onehot), y_dev_onehot.shape))

    # Build the machine learning model as follows: 
    # CNN (4 layer) --> LSTM (2 layer) --> Dense (1 layer) --> Softmax --> Cross entropy loss
    #
    # 1. CNN is suitable for character level sequence (or time series) classification. It helps to extract relevant patterns from the sequences along the feature and time dimensions.
    # 2. LSTM helps to recognise sequential information
    # 3. Fully connected Dense layer with Softmax reduces the output to 12 desired class probability distribution
    # 4. Cross entropy loss is used as the cost for this multiclass classification problem.
    
    print('Building the model')
    kernel_length = [ 5,   3,   3,   3]
    filter_num    = [64,  64, 128, 128]
    pool_length   = [ 2,   2,   2,   2]
    x_input = Input(shape=(Tx,), dtype='int64')
    # The one-hot encoded vectors for each input string is built on the fly using the Lambda layer
    # Desired shape of input data X = (m, Tx, n_v)
    x_one_hot = Lambda(one_hot, output_shape=one_hot_outshape, arguments={'numOfVocab': 26})(x_input)
    x_conv = x_one_hot
    for ii in range(len(filter_num)):
        x_conv = Conv1D(filters=filter_num[ii],
                        kernel_size=kernel_length[ii],
                        padding='same')(x_conv)
        x_conv = Activation('relu')(x_conv)
        x_conv = MaxPooling1D(pool_size=pool_length[ii])(x_conv)

    x_lstm1 = Bidirectional(LSTM(128, return_sequences=True))(x_conv)
    x_lstm1_dropout = Dropout(0.3)(x_lstm1)
    x_lstm2 = LSTM(128, input_shape = (None, 256), return_sequences=False)(x_lstm1_dropout)
    x_lstm2_dropout = Dropout(0.3)(x_lstm2)
    x_out = Dense(n_c, activation='softmax')(x_lstm2_dropout)
    model = Model(inputs=x_input, outputs=x_out, name='textClassifier')
    # Check model summary
    model.summary()

    # Define an optimizer and compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Save the model at regular epoch ntervals.
    checkpoint = keras.callbacks.ModelCheckpoint('./cnn_lstm'+'-{epoch:02d}-{val_acc:.2f}.hdf5',
                                                 monitor='val_loss', 
                                                 verbose=1, 
                                                 save_best_only=False, 
                                                 save_weights_only=False, 
                                                 mode='auto', 
                                                 period=10)
    
    # Fit the model
    print('Fitting the model now')
    model.fit(x_train, y_train_onehot,
              validation_data=(x_dev, y_dev_onehot),
              batch_size=256, 
              epochs=130,
              shuffle=True,
              callbacks=[checkpoint],
              class_weight=class_weight_dict)
    
    # Save model to continue later
    model.save('./cnn_lstm.hdf5')

    # We have already trained the above model over 180 epochs, which yielded a 
    # training set accuracy of ~99% and validation set accuracy of ~87%.
    # Hence, we now load the trained model containing the trained parameters.

    #Load saved model 
    #Our saved file is named "cnn_lstm-180-0.87.hdf5"
    model = load_model('./cnn_lstm-180-0.87.hdf5')
    
    # First check accuracy on training and validations set
    _, acc_train = model.evaluate(x_train, y_train_onehot)
    _, acc_dev = model.evaluate(x_dev, y_dev_onehot)
    print('Train set accuracy: {}'.format(acc_train))
    print('Validation set accuracy: {}'.format(acc_dev))
    
    #Read in the test set
    x_test = readIn('./xtest.txt')
    x_test = listOfStr_to_array(x_test, Tx, lambda x: char_to_ix[x])
    # Check type and shape x_test
    print('x_test is of type {} and of shape {}'.format(type(x_test), x_test.shape))
    
    #Make predictions
    pred = model.predict(x_test)
    #Write predictions to file
    pred_labels = np.argmax(pred,axis=1)
    write2file('ytest.txt',pred_labels)

    return 0

if __name__ == "__main__":
    main()
