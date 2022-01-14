import glob
import pydicom
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import *
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, \
    Activation, add, concatenate, GlobalAveragePooling2D, Reshape, Dense, Permute, multiply
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K

###### de-identifying of CT images ###################
def De_identification(p1,p2):
    """
    p1 = "Path for reading CTs"
    p2 = "Path for saving de-identified CTs"
    """
    images = np.array(glob.glob(p1 + "/*.dcm"), dtype=np.str)

    for i in range(images.shape[0]):
        ds = pydicom.read_file(images[i])
        # delete un-desired attributes
        del ds.PatientName
        del ds.InstitutionAddress
        del ds.InstitutionName
        del ds.OperatorsName
        del ds.PatientBirthDate
        del ds.PatientID
        del ds.StudyID
        del ds.AcquisitionNumber
        del ds.StationName
        del ds.ReferringPhysicianName
        if i < 9:
            z = "00"
        elif i > 98:
            z = ""
        else:
            z = "0"
        ds.save_as(p2 + "im" + z + str(i + 1) + '.dcm')

##################################################################################################
# Synthetic data generation method
##################################################################################################
def significant_lung_regions(cts, lungs, thresh=0.03):
    """
    selecting normal CT images with (lung area)/(image area) > thresh
    Lungs : binary lung masks for normal ct images
    cts : normal ct images
    """
    cts_normal = []
    lungs_normal = []

    for i in range(lungs.shape[0]):
        if np.sum(lungs[i, ...]) / (512 * 512) > thresh:
            cts_normal.append(cts[i, ...])
            lungs_normal.append(lungs[i, ...])

    return np.array(cts_normal), np.array(lungs_normal)
###

def preprocessing(ct, lung):
    """
    Normalizing ct images and removing non-lung regions
    input:
    ct : input ct images
    lung : lung mask for each CT image

    output: normalized lung regions
    """
    normalized_cts = []
    for i in range(ct.shape[0]):
        im = ct[i,...]
        lng = lung[i,...]
        im = (im  - im.min())/(im.max() - im.min())
        im = im * lng
        normalized_cts.append(im)

    return np.array(normalized_cts)


def synthetic_image(cts_infected, lungs_infected, infection_mask, cts_normal, lungs_normal):
    """
    cts_infected, lungs_infected : infected ct images and corresponding lung masks
    infection_mask : lesion masks for infected cts
    cts_normal, lungs_normal : normal ct images and corresponding lung masks
    """
    cts_normal, lungs_normal = significant_lung_regions(cts_normal, lungs_normal)

    ct_infected_normalized = preprocessing(cts_infected, lungs_infected)
    ct_normal_normalized = preprocessing(cts_normal, lungs_normal)

    # extracting infection regions
    inf_region = ct_infected_normalized * infection_mask

    # inversing infection masks
    y_inv = 1 - infection_mask

    # inserting infection regions into normal cts
    ct_normal_plus_infection = ct_normal_normalized * y_inv + inf_region

    # removing infection regions outside the lung area
    ct_synthetic = ct_normal_plus_infection * lungs_normal

    # generating synthesized masks
    infection_mask_synthetic = infection_mask * lungs_normal

    # Selecting synthetic ct images with infection_rate greater than the threshold
    thresh = 0.01
    ct_synthetic_sig = []
    infection_mask_synthetic_sig = []
    for i in range(ct_synthetic.shape[0]):
        infection_rate = np.sum(infection_mask_synthetic[i, ...]) / np.sum(lungs_normal[i, ...])
        if infection_rate > thresh:
            ct_synthetic_sig.append(ct_synthetic[i, ...])
            infection_mask_synthetic_sig.append(infection_mask_synthetic[i, ...])
    ct_synthetic_sig = np.array(ct_synthetic_sig)
    infection_mask_synthetic = np.array(infection_mask_synthetic_sig)

    # Normalizing synthetic ct images
    ct_synthetic_normalized = []
    for i in range(ct_synthetic_sig.shape[0]):
        im = ct_synthetic_sig[i, ...]
        im = (im - im.min()) / (im.max() - im.min())
        ct_synthetic_normalized.append(im)
    ct_synthetic_normalized = np.array(ct_synthetic_normalized)

    return ct_synthetic_normalized, infection_mask_synthetic
################## Segmentation Network ##########################
### loss functions
def dice_coefficient(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def wighted_bce(y_true, y_pred, w1):
    weights = (y_true * w1) + 1.
    bce = K.binary_crossentropy(y_true, y_pred)
    weighted_bce = K.mean(bce * weights)
    return weighted_bce

def tversky(y_true, y_pred, smooth = 0.000001):
    alpha = 0.7
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return ((true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth))

def tversky_loss(y_true, y_pred):
    return (1 - tversky(y_true, y_pred))

def focal_tversky_loss(y_true, y_pred):
    gamma = 0.75
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), gamma)

def bce_ftvsky_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + focal_tversky_loss(y_true, y_pred)
    return loss

def weighted_bce_ftvsky(y_true, y_pred, w1):
    loss = wighted_bce(y_true, y_pred, w1) + focal_tversky_loss(y_true, y_pred)
    return loss

### Network
def res_block_unit1(X, filters):
    e1 = X

    x = Conv2D(filters, kernel_size=(3, 3), padding='same', strides=(2,2),
               kernel_initializer = 'he_normal', trainable=True)(e1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size=(3, 3), dilation_rate=2, padding='same',
               strides=(1,1), kernel_initializer = 'he_normal', trainable=True)(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = squeeze_excite_block(x, ratio=16)

    shortcut = Conv2D(filters, (1, 1), strides=(2,2), trainable=True)(X)
    shortcut = BatchNormalization()(shortcut)

    x = add([shortcut, x])

    return x, e1

def res_block_unit2(X, filters):
    e1 = X
    x = Conv2D(filters, kernel_size=(3, 3),dilation_rate=1, padding='same',
               strides=(1,1), kernel_initializer = 'he_normal', trainable=True)(e1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size=(3, 3),dilation_rate=2,padding='same',
               strides=(1,1), kernel_initializer = 'he_normal', trainable=True)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = squeeze_excite_block(x, ratio=16)

    x = add([e1, x])

    return x

def decoder_block(X, filters, cn):
    x = X

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters, kernel_size=(3, 3), padding='same',
               strides=(1, 1), kernel_initializer = 'he_normal', trainable=True)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    cn = Conv2D(filters, kernel_size=(1, 1), padding='same', activation='relu',
               strides=(1, 1), kernel_initializer='he_normal', trainable=True)(cn)
    x = concatenate([x, cn])
    x = Conv2D(filters, kernel_size=(3, 3), dilation_rate=2, padding='same',
               strides=(1, 1), kernel_initializer='he_normal', trainable=True)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x, ratio=16)
    return x


# For the squeeze_excite_block, we used the code from:
# https://github.com/titu1994/keras-squeeze-excite-network

def squeeze_excite_block(tensor, ratio=16):
    init = tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x + init

def CPB(X, filters):

    x0 = Conv2D(filters, kernel_size=(1, 1),dilation_rate=1, padding='same', kernel_regularizer= 'l2',
               strides=(1,1), kernel_initializer = 'he_normal')(X)

    x1 = Conv2D(filters, kernel_size=(3, 3),dilation_rate=1, padding='same', kernel_regularizer= 'l2',
               strides=(1,1), kernel_initializer = 'he_normal')(X)

    x2 = Conv2D(filters, kernel_size=(3, 3), dilation_rate=2, padding='same', kernel_regularizer= 'l2',
                strides=(1, 1), kernel_initializer='he_normal')(X)

    x3 = Conv2D(filters, kernel_size=(3, 3), dilation_rate=4, padding='same', kernel_regularizer= 'l2',
                strides=(1, 1), kernel_initializer='he_normal')(X)

    x4 = Conv2D(filters, kernel_size=(3, 3), dilation_rate=8, padding='same', kernel_regularizer= 'l2',
                strides=(1, 1), kernel_initializer='he_normal')(X)

    x = add([x0,x1,x2,x3,x4])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def COVID_Rate(inputs, filters=None):
    if filters is None:
        filters = [32, 64, 128, 256, 512]
    x = Conv2D(filters[0], (7,7), strides=(1,1), dilation_rate=1,
               padding='same', kernel_initializer = 'he_normal', trainable=True)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #s0
    x, r0 = res_block_unit1(x, filters[0])
    x = res_block_unit2(x, filters[0])
    #s1
    x, r1 = res_block_unit1(x, filters[1])
    x = res_block_unit2(x, filters[1])
    #s2
    x, r2 = res_block_unit1(x, filters[2])
    x = res_block_unit2(x, filters[2])
    # s3
    x, r3 = res_block_unit1(x, filters[3])
    x = res_block_unit2(x, filters[3])
    x = CPB(x, filters[3])
    # encoder4
    x = decoder_block(x, filters[4],r3)
    # encoder3
    x = decoder_block(x, filters[3], r2)
    # encoder2
    x = decoder_block(x, filters[2], r1)
    # encoder1
    x = decoder_block(x, filters[1], r0)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

model = COVID_Rate(inputs=Input(shape=(512,512,1)))
opt = Adam(learning_rate=0.001, decay= 0.0004, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer= opt , loss= weighted_bce_ftvsky , metrics=['accuracy', dice_coefficient])

##################################################
# Unsupervise Enhancement- Certainty index
##################################################
def Certainty_index(predicted_masks):
    """
    predicted_masks : predicted masks on external test set that we are blind to their ground-truth infection masks
    """
    certainty_indexes = []
    thresh = 0.3  # cut-off probability
    for prb in predicted_masks:  # prb: predictred probability for each pixel of a predicted infection mask
        certainty_map = np.where(prb > thresh, prb, 0)
        num = np.count_nonzero(certainty_map)  # number of pixels predicted as the infection class
        if num > 0:
            certainty_index = np.sum(certainty_map) / num
            certainty_indexes.append(certainty_index)
        else:
            certainty_indexes.append(0)
    certainty_indexes = np.array(certainty_indexes)

    return certainty_indexes



