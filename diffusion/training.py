import numpy as np
import tensorflow as tf
import Unet
import sys
import os
import pickle, gzip
import matplotlib.pyplot as plt

sys.path.append('../cv_project')
from preprocess_data import load_smallnorbMine

Image_size = 32
learning_rate = 1e-3
weight_decay = 1e-4

# architecture
embedding_dims = 32
embedding_max_frequency = 1000.0
widths = [32, 64, 128]
block_depth = 2

# optimization
batch_size = 60

#This process the data for smallNorb
def processSmallNorbDataSet(train_images, test_images):
    nums, width, height, channels = train_images.shape[0], train_images.shape[1], train_images.shape[2], 1
    nums_test = test_images.shape[0]
    x_train_org = (np.concatenate((train_images[:, :, :, 0], train_images[:, :, :, 1])))
    x_train_org = x_train_org.reshape((nums*2, width, height, channels)).astype('float32')

    x_test_org = (np.concatenate((test_images[:, :, :, 0], test_images[:, :, :, 1])))
    x_test_org = x_test_org.reshape((nums_test * 2, width, height, channels)).astype('float32')

    preProcess = tf.keras.Sequential(
        [
            tf.keras.layers.Resizing(Image_size, Image_size),
        ]
    )

    x_train = preProcess(x_train_org)
    x_test = preProcess(x_test_org)

    x_train = (x_train/ 127.5 ) - 1.0

    return x_train, x_test

#Custom loss function described in the report
def custom_loss_function(noise, noise_pred):
    return tf.math.reduce_mean((noise - noise_pred) ** 2)

#Will train and then display a 6 images.

def training(load_from_file = False, verbose = False, dim_in = Image_size, dim_out = 1, epochs = 10):
    # ProcessingData to a standardize set.
    (train_images, train_labels), (test_images, test_labels) = load_smallnorbMine()
    x_train, x_test = processSmallNorbDataSet(train_images, test_images)

    model = Unet.DiffusionModel(Image_size, widths, block_depth)

    x_train_sample = x_test[:16]

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.2, wspace=0.1)
    for i, ax in enumerate(axes.flat):
        # Just show the left image
        image = (x_train_sample[i]+1.0)*127.5
        # Show image
        ax.imshow(image, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()



    # Specify the names of the save files
    save_name = os.path.join('saved', 'Unet_res_2')
    net_save_name = save_name + '_cnn_net.keras'
    checkpoint_save_name = save_name + '_cnn_net.chk.keras'
    history_save_name = save_name + '_cnn_net.hist'

    if load_from_file and os.path.isfile(net_save_name):
        # ***************************************************
        # * Loading previously trained neural network model *
        # ***************************************************

        # Load the model from file
        if verbose:
            print("Loading neural network from %s..." % net_save_name)
        net = tf.keras.models.load(net_save_name)
        # Load the training history - since it should have been created right after
        # saving the model
        if os.path.isfile(history_save_name):
            with gzip.open(history_save_name) as f:
                history = pickle.load(f)
        else:
            history = []
    else:
        adam = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        #Using legacy here as works best with m1 macs.
        model.compile(
            optimizer=adam,
            loss=custom_loss_function,
        )

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_save_name,
            monitor='i_loss',
            mode='min',
            save_best_only=True)

        # calculate mean and variance of training dataset for normalization
        #  tf.keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images),
        # Training
        train_info = model.fit(x_train, epochs=epochs, shuffle=True,
                             callbacks=[model_checkpoint_callback,tf.keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images)], batch_size=batch_size)

        model.plot_images()
        model.plot_images()
        model.plot_images()
        model.plot_images()
        model.plot_images()

        model.plot_images()


        #load the best model
        # load the best model and generate images
        model.plot_images()
        # Save the entire model to file
        print("Saving neural network to %s..." % net_save_name)
        model.save(net_save_name)

        model.load_weights(net_save_name)
        # Save training history to file
        history = train_info.history
        with gzip.open(history_save_name, 'w') as f:
            pickle.dump(history, f)


        return model


if __name__ == "__main__":
    training(load_from_file=False, verbose=False, dim_in=Image_size, dim_out=1, epochs=2)