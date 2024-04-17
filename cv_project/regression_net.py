import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pickle, gzip
import preprocess_data as pp
from keras.layers import Conv2D, BatchNormalization, Activation, AveragePooling2D, Add, RandomFlip, RandomRotation, RandomZoom

Image_size = 64

#####
# Loads the small_norb dataset and 
# concatates the channels together.
# Resize, Resales and Centre crops the images.
# Returns train and test as tf dataframes.
#####
def processSmallNorbDataSet(train_images, y_hat_train,test_images, y_hat_test, resize =False, data_aug=False,batch_size = 32):
    nums, width, height, channels = train_images.shape[0], train_images.shape[1], train_images.shape[2], 1
    nums_test = test_images.shape[0]
    x_train_org = (np.concatenate((train_images[:, :, :, 0], train_images[:, :, :, 1])))
    x_train_org = x_train_org.reshape((nums*2, width, height, channels))
    y_hat_train = np.concatenate((y_hat_train, y_hat_train))
    y_hat_train = y_hat_train/9.0

    x_test_org = (np.concatenate((test_images[:, :, :, 0], test_images[:, :, :, 1])))
    x_test_org = x_test_org.reshape((nums_test * 2, width, height, channels))
    y_hat_test = np.concatenate((y_hat_test, y_hat_test))
    y_hat_test = y_hat_test/9.0

    #Keras Functional api for preprocssing the image data.
    preProcess = tf.keras.Sequential(
        [
            tf.keras.layers.Resizing(Image_size, Image_size),
            tf.keras.layers.Rescaling(scale= 1.0/255),
            tf.keras.layers.CenterCrop(Image_size, Image_size)
        ]
    )

    x_train = preProcess(x_train_org)
    x_test = preProcess(x_test_org)

    return x_train, y_hat_train, x_test, y_hat_test

####
# Using keras functional API we have the building block of the model, the resnet block based off of the resnet block from resnet50. 
# See "Deep Residual Learning for Image Recognition" by He et al. for more details. Resnet block is both the indentiy block 
# and the cnn block described in the paper, if first block is == to True then it is the cnn block.
###
def resnet_block(inputs, filters, strides=(1, 1), first_block = False, drop_out = 0):
    # Save the input tensor for the skip connection
    x_shortcut = inputs

    f1, f2, f3 = filters

    # First convolutional layer
    x = Conv2D(filters=f1, kernel_size=(1, 1), strides=strides, padding='same')(inputs)
    x = BatchNormalization(axis = 3)(x)
    x = Activation('relu')(x)

    # Second convolutional layer
    x = Conv2D(filters=f2, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis = 3)(x)
    x = Activation('relu')(x)

    # Third component
    x = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1,1), padding='valid')(x)
    x = BatchNormalization(axis = 3)(x)


    if first_block: # i.e the bloack isn't the identiy block
        x_shortcut = Conv2D(filters=f3, kernel_size=(1,1), strides = strides, padding='valid')(x_shortcut)
        x_shortcut = BatchNormalization(axis = 3)(x)


    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)
    if drop_out != 0:
        x = tf.keras.layers.SpatialDropout2D(drop_out)(x)

    return x

####
# Loads saved model, if the model is not been trained (or saved), the function will train the model as per the parameters.
####
def resnet(input_shape, num_classes, data_aug = False, reg_wdecay_beta = 0, drop_out = 0):

    data_augmentation = tf.keras.Sequential(
        [
            RandomFlip("horizontal_and_vertical"),
            RandomRotation(0.1),
            RandomZoom(0.1)
        ])

    # Define input tensor
    inputs = tf.keras.layers.Input(shape=input_shape)

    #Normalzation

    # Zero-Padding
    x = tf.keras.layers.ZeroPadding2D((3, 3))(inputs)

    if data_aug:
        x = data_augmentation(x)

    # First convolutional layer
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis = 3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((3,3), strides=(2,2))(x)


    # First ResNet Block # Note: downsampling here is already done to don't specify a stride length
    x = resnet_block(x, filters=[64,64,256], first_block = True, drop_out = drop_out )
    x = resnet_block(x, filters=[64, 64, 256], drop_out = drop_out)
    #x = resnet_block(x, filters=[64, 64, 256], drop_out = drop_out )

    # Second ResNetBlock
    x = resnet_block(x, filters=[128, 128, 512], strides=(2,2), first_block=True,drop_out = drop_out )
    x = resnet_block(x, filters=[128, 128, 512],drop_out = drop_out )
    x = resnet_block(x, filters=[128, 128, 512],drop_out = drop_out )
    #x = resnet_block(x, filters=[128, 128, 512],drop_out = drop_out )

    # Third ResNet Block
    x = resnet_block(x, filters=[256, 256, 1024], strides=(2, 2), first_block=True,drop_out = drop_out )
    x = resnet_block(x, filters=[256, 256, 1024],drop_out = drop_out )
    x = resnet_block(x, filters=[256, 256, 1024],drop_out = drop_out )
    #x = resnet_block(x, filters=[256, 256, 1024],drop_out = drop_out )
    #x = resnet_block(x, filters=[256, 256, 1024],drop_out = drop_out )
    #x = resnet_block(x, filters=[256, 256, 1024],drop_out = drop_out )

    # Fourth ResNet Block
    x = resnet_block(x, filters=[512, 512, 2048], strides=(2, 2), first_block=True,drop_out = drop_out )
    x = resnet_block(x, filters=[512, 512, 2048],drop_out = drop_out )
    #x = resnet_block(x, filters=[512, 512, 2048],drop_out = drop_out )

    if reg_wdecay_beta > 0:
        reg_wdecay = tf.keras.regularizers.l2(reg_wdecay_beta)
    else:
        reg_wdecay = None

    # Global average pooling and dense layers
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units = 1000, activation='relu')(x)
    x = tf.keras.layers.Dense(units=1)(x)

    # Define the model
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    return model

def regressionA1_trained_cnn(load_from_file = False, verbose = True,reg_wdecay_beta = 0, reg_dropout_rate = 0,data_aug=False, early_stop = 0):

    class_names = pp.category_labels

    #Loaded in the data
    (train_images, train_labels), (test_images, test_labels) = pp.load_smallnorbMine()

    #Using the angle.
    y_hat_train = train_labels[:,3]
    y_hat_test = test_labels[:,3]

    #
    x_train, y_hat_train, x_test, y_hat_test = processSmallNorbDataSet(train_images, np.squeeze(y_hat_train), test_images,np.squeeze(y_hat_test), data_aug)

    # Create 'saved' folder if it doesn't exist
    if not os.path.isdir("saved"):
        os.mkdir('saved')

    # Specify the names of the save files
    save_name = os.path.join('saved', 'regressionA1ResNet_rwd%.1e_rdp%d_daug%d%d' % (reg_wdecay_beta,reg_dropout_rate,int(data_aug),early_stop))
    net_save_name = save_name + '_cnn_net.keras'
    checkpoint_save_name = save_name + '_cnn_net.chk.keras'
    history_save_name = save_name + '_cnn_net.hist'
    # Show 16 train images with the corresponding labels

    if verbose:
        x_train_sample = x_test[:16] *255
        y_hat_train_sample = y_hat_test[:16]
        print(y_hat_train_sample)

        fig, axes = plt.subplots(3, 3, figsize=(8, 8))
        fig.subplots_adjust(hspace=0.2, wspace=0.1)
        for i, ax in enumerate(axes.flat):
            # Just show the left image
            image = x_train_sample[i]
            label = (y_hat_train_sample[i] * 45) + 30
            # Show image
            ax.imshow(image, cmap='gray')
            ax.text(0.5, -0.12, f'Label: {label}', ha='center',
                    transform=ax.transAxes, color='black')
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()
    n_classes = len(class_names)

    if load_from_file and os.path.isfile(net_save_name):
        # ***************************************************
        # * Loading previously trained neural network model *
        # ***************************************************

        # Load the model from file
        if verbose:
            print("Loading neural network from %s..." % net_save_name)
        net = tf.keras.models.load_model(net_save_name)

        # Load the training history - since it should have been created right after
        # saving the model
        if os.path.isfile(history_save_name):
            with gzip.open(history_save_name) as f:
                history = pickle.load(f)
        else:
            history = []
    else:
        # ************************************************
        # * Creating and training a neural network model *
        # ************************************************

        net = resnet(input_shape= (64,64,1), num_classes= n_classes, reg_wdecay_beta=reg_wdecay_beta, data_aug=True,drop_out =  reg_dropout_rate)
        adam = tf.keras.optimizers.SGD(learning_rate=0.001, weight_decay=reg_wdecay_beta, momentum=0.9)
        # Define training regime: type of optimiser, loss function to optimise and type of error measure to report during
        # training
        net.compile(optimizer=adam,
                    loss='mse',
                    metrics=['mae'])

        # Training callback to call on every epoch -- evaluates
        # the model and saves its weights if it performs better
        # (in terms of accuracy) on validation data than any model
        # from previous epochs
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_save_name,
            monitor='loss',
            mode='min',
            save_best_only=True)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stop)
        
        callbacks_vals = [model_checkpoint_callback]

        if early_stop > 0:
            callbacks_vals.append(early_stopping)

        #Training
        train_info = net.fit(x_train, y_hat_train, validation_split=0.33 ,  epochs=5, shuffle=True,
                                    callbacks=callbacks_vals)

        # Load the weights of the best model
        print("Loading best saved weights from %s..." % checkpoint_save_name)
        net.load_weights(checkpoint_save_name)

        # Save the entire model to file
        print("Saving neural network to %s..." % net_save_name)
        net.save(net_save_name)

        # Save training history to file
        history = train_info.history
        with gzip.open(history_save_name, 'w') as f:
            pickle.dump(history, f)

    # *********************************************************
    # * Training history *
    # *********************************************************

    # Plot training and validation accuracy over the course of training
    if verbose and history != []:
        fh = plt.figure()
        ph = fh.add_subplot(111)
        ph.plot(history['mae'], label='mae')
        ph.plot(history['val_mae'], label = 'val_mae')
        ph.set_xlabel('Epoch')
        ph.set_ylabel('MAE')
        ph.set_ylim([0, 1])
        ph.legend(loc='lower right')
        plt.show()

    # *********************************************************
    # * Evaluating the neural network model within tensorflow *
    # *********************************************************

    if verbose:
        loss_train, accuracy_train = net.evaluate(x_train, y_hat_train, verbose=0)
        loss_test, accuracy_test = net.evaluate(x_test,y_hat_test,verbose=0)

        print("Train Mae (tf): %.2f" % accuracy_train)
        print("Test Mae  (tf): %.2f" % accuracy_test)
        print(net.summary())

        # Compute output for 16 test images
        pred_index = np.random.randint(low = 0, high = len(y_hat_test)-1, size = 9)

        fig, axes = plt.subplots(3, 3, figsize=(8, 8))
        fig.subplots_adjust(hspace=0.2, wspace=0.1)
        for i, ax in enumerate(axes.flat):
            # Just show the left image
            index = pred_index[i]
            y_test_pred = net.predict(x_test[index])
            image = x_test[index]*255
            label = (y_hat_test[index] * 45) + 30
            pred = (y_test_pred[0] * 45 ) + 30

            # Show image
            ax.imshow(image, cmap='gray')
            ax.text(0.5, -0.12, f'Label: {label}, Pred: {pred}', ha='center',
                    transform=ax.transAxes, color='black')
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()
    return net

if __name__ == "__main__":
   regressionA1_trained_cnn(load_from_file=False, verbose=True ,data_aug=False, reg_wdecay_beta = 0, reg_dropout_rate =0, early_stop= 5)
