from typing import List
import numpy as np
import pickle, gzip
import struct
import os



#This file is a combination with the reference above and load_small norb as tf datasets didn't work
#for me. Basically we are using bit encoding from the link that you sent my from NYU
## Using bit encoding 
##

PREFIXES = {
    'train': 'smallnorb-5x46789x9x18x6x2x96x96-training-',
    'test': 'smallnorb-5x01235x9x18x6x2x96x96-testing-',
}

data_path = "../data/"

FILE_TYPES = ['info', 'cat', 'dat']
numpy_save = 'smallnorb.data'

map_magic_number_to_data_type = {
    '1e3d4c55': np.uint8,
    '1e3d4c54': np.int32,
}

category_labels = ['animal', 'human', 'airplane', 'truck', 'car']

# helper function to read int from file
def read_int(f):
    num, = struct.unpack('i', f.read(4))
    return num

loaded_data = {}

## Function from: https://www.kaggle.com/code/leshabirukov/small-norb-load/notebook
def processData():
    for dataset, prefix in PREFIXES.items():
        for filetype in FILE_TYPES:
            filename = data_path + prefix + filetype + ".mat"
            print('Reading {}'.format(filename))

            file_loc = os.path.join(filename)
            with open(file_loc, 'rb') as f:
                # Read the magic_num, convert it to hexadecimal, and look up the data_type
                raw_magic_num = read_int(f)
                magic_num = format(raw_magic_num, '02x')
                data_type = map_magic_number_to_data_type[magic_num]
                print('dtype', data_type)

                # Read how many dimensions to expect
                ndim = read_int(f)

                # Read at least 3 ints, or however many ndim there are
                shape = [
                    read_int(f)
                    for i in range(max(ndim, 3))
                ]
                # But in case ndims < 3, take at most n_dim elements
                shape = shape[:ndim]
                print('shape', shape)

                # Now load the actual data!
                loaded_data[(dataset, filetype)] = np.fromfile(
                    f,
                    dtype=data_type,
                    count=np.prod(shape)
                ).reshape(shape)

def load_smallnorbMine():

    processData()

    N = 24300
    train_images = np.zeros((N, 96, 96, 2)).astype('uint8')
    train_labels = np.zeros((N, 5)).astype('uint8')

    test_images = np.zeros((N, 96, 96, 2)).astype('uint8')
    test_labels = np.zeros((N, 5)).astype('uint8')


    train_images_pre = np.array(loaded_data[('train','dat')])
    test_images_pre = np.array(loaded_data[('test','dat')])

    for n in range(N):
        left_train = train_images_pre[n,0,:,:]
        right_train = train_images_pre[n,1,:,:]
        left_test = test_images_pre[n,0,:,:]
        right_test = test_images_pre[n,1,:,:]

        train_images[n, :, :, 0] = left_train
        train_images[n, :, :, 1] = right_train
        test_images[n, :, :, 0] = left_test
        test_images[n, :, :, 1] = right_test


    train_labels_info = np.array(loaded_data[('train','info')]).reshape(N,4)
    train_labels_pre = np.array(loaded_data[('train','cat')]).reshape(N,)

    test_labels_info = np.array(loaded_data[('test', 'info')]).reshape(N, 4)
    test_labels_pre = np.array(loaded_data[('test', 'cat')]).reshape(N,)

    # 0 instance of toy
    # 1 azimuth angles
    # 2 labels of toys
    # 3 angles of the toys for the elevation
    # 4 the specific lighting.


    train_labels[:,0] = train_labels_info[:,0].astype('uint8')
    train_labels[:, 1] = train_labels_info[:, 2].astype('uint8')
    train_labels[:, 2] = train_labels_pre.astype('uint8')
    train_labels[:,3] = train_labels_info[:,1].astype('uint8')
    train_labels[:,4] = train_labels_info[:,3].astype('uint8')

    test_labels[:, 0] = test_labels_info[:, 0].astype('uint8')
    test_labels[:, 1] = test_labels_info[:, 2].astype('uint8')
    test_labels[:, 2] = test_labels_pre.astype('uint8')
    test_labels[:, 3] = test_labels_info[:, 1].astype('uint8')
    test_labels[:, 4] = test_labels_info[:, 3].astype('uint8')

    return (train_images, train_labels),(test_images,test_labels)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load the data in the numpy format
    (train_images, train_labels), (test_images, test_labels) = load_smallnorbMine()

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.2, wspace=0.1)
    for i, ax in enumerate(axes.flat):
        # Just show the left image
        image = train_images[i, :, :, 0]
        # Fetch category
        category_str = category_labels[train_labels[i, 2]]
        # Get the azimuth (* by 10 to get azimuth according to cs.nyc websidte)
        azimuth = train_labels[i, 1] * 10
        # Get the elevation
        elevation = train_labels[i, 3] * 5 + 30
        # Get the lighting
        lighting = train_labels[i, 4]

        # Show image
        ax.imshow(image, cmap='gray')
        ax.text(0.5, -0.12, f'{category_str} (el={elevation},az={azimuth},lt={lighting})', ha='center',
                transform=ax.transAxes, color='black')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
