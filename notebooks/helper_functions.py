import numpy as np
import matplotlib.pyplot as plt


def import_data(path=None, num_samples=None):
    """ Imports scintillator data as numpy arrays.
    Used together with analysis repository which has a strict folder
    structure.

    param path: Path to datafile

    param num_samples:  How many samples to include. With large files,
                        memory might become an issue when loading full file.
                        If specified, the returned data will be a random,
                        balanced selection of data from the full dataset.

    param scaling:  Whether or not to scale the image data to 0-1 interval.
                    Defaults to False.


    returns:    dictionary of data where each filenames are keys and each
                key,value pair contains dictionary of the data in the file,
                separated into 'energies', 'positions', 'images', 'labels'.
    """

    # Temporary initialization of arrays-to-be
    images = []
    energies = []
    positions = []
    labels = []

    # Read line by line to alleviate memory strain when files are large
    # The first 256 values in each row correspond to the 16x16 detector image,
    # the last 6 values correspond to Energy1, Xpos1, Ypos1, Energy2, Xpos2,
    # Ypos2.

    with open(path, "r") as infile:
        for line in infile:
            line = np.fromstring(line, sep=' ')
            image = line[:256]
            energy = np.array((line[256], line[259]))
            pos = np.array((line[257], line[258], line[260], line[261]))

            # Set label for the events. If Energy2 is 0 it is a single
            # event. Any other values corresponds to a double event.
            # We label single events as type 0, and doubles as type 1
            if energy[1] == 0:
                label = 0
            else:
                label = 1

            images.append(image)
            energies.append(energy)
            positions.append(pos)
            labels.append(label)

    # Convert lists to numpy arrays and reshape them to remove the added axis
    # conversion.
    images = np.array(images)
    energies = np.array(energies)
    positions = np.array(positions)
    labels = np.array(labels)
    return images, energies, positions, labels


def normalize_image_data(images):
    """ Takes an imported set of images and normalizes values to between
    0 and 1 using min-max scaling across the whole image set.
    """
    img_term = np.amax(images) - np.amin(images)
    img_mean = np.mean(images)
    images = (images - img_mean) / img_term
    return images

def plot_history(history):
    """ Plots loss, val_loss, accuracy, and val_accuray as two plots
    side-by-side.
    """
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    num_epochs = len(history.history['loss'])
    ax[0].plot(history.history['loss'], label='training')
    ax[0].plot(history.history['val_loss'], label='validation')
    ax[0].set_title("Model loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_xticks(np.arange(num_epochs))
    ax[0].legend()

    ax[1].plot(history.history['accuracy'], label='training')
    ax[1].plot(history.history['val_accuracy'], label='validation')
    ax[1].set_title("Model accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_xticks(np.arange(num_epochs))
    ax[1].legend()
    return fig, ax