# This code is modified from CS231n.
import json
import numpy as np
import h5py
import urllib.request, urllib.error, urllib.parse, tempfile, os
from scipy.misc import imread

DATA_DIR = 'data/coco_captioning'  # define the dataset path


def load_coco_dataset(data_dir=DATA_DIR, PCA_features=True, max_train=None):
    """
    Load Microsoft COCO dataset.
    Arguments:
        data_dir: path to the dataset
        PCA_features: whether use PCA features
        max_train: max number of training data if only a subset is needed
    Outputs:
        data: dictionary containing different datasets with their names
    """
    data = {}
    caption_file = os.path.join(data_dir, 'coco2014_captions.h5')
    with h5py.File(caption_file, 'r') as f:  # read caption file with h5py
        for k, v in f.items():
            data[k] = np.asarray(v)

    # extract training features
    if PCA_features:
        train_feature_file = os.path.join(data_dir, 'train2014_vgg16_fc7_pca.h5')
    else:
        train_feature_file = os.path.join(data_dir, 'train2014_vgg16_fc7.h5')
    with h5py.File(train_feature_file, 'r') as f:
        data['train_features'] = np.asarray(f['features'])

    # extract validation features
    if PCA_features:
        val_feature_file = os.path.join(data_dir, 'val2014_vgg16_fc7_pca.h5')
    else:
        val_feature_file = os.path.join(data_dir, 'val2014_vgg16_fc7.h5')
    with h5py.File(val_feature_file, 'r') as f:
        data['val_features'] = np.asarray(f['features'])

    # extract index-to-word and word-to-index into dictionary
    dict_file = os.path.join(data_dir, 'coco2014_vocab.json')
    with open(dict_file, 'r') as f:
        dict_data = json.load(f)
        for k, v in dict_data.items():
            data[k] = v

    # read image files from website, note that some of them might not be available for now
    train_url_file = os.path.join(data_dir, 'train2014_urls.txt')  # this file includes urls for the training images
    with open(train_url_file, 'r') as f:
        train_urls = np.asarray([line.strip() for line in f])
    data['train_urls'] = train_urls

    val_url_file = os.path.join(data_dir, 'val2014_urls.txt')  # this file includes urls for the validation images
    with open(val_url_file, 'r') as f:
        val_urls = np.asarray([line.strip() for line in f])
    data['val_urls'] = val_urls

    # Maybe subsample the training data
    if max_train is not None:
        num_train = data['train_captions'].shape[0]
        mask = np.random.randint(num_train, size=max_train)
        data['train_captions'] = data['train_captions'][mask]
        data['train_image_idxs'] = data['train_image_idxs'][mask]

    return data


def sample_coco_minibatch(data, batch_size=100, split='train'):
    """
    Sample a small amount of data.
    Arguments:
        data: loaded dataset from COCO
        batch_size: int for batch size
        split: string of either 'train' or 'val' indicating training/validation set
    Outputs:
        captions: ground truth captions of the images
        image_features: features of the images
        urls: image urls for image display
    """
    split_size = data['%s_captions' % split].shape[0]
    mask = np.random.choice(split_size, batch_size)
    captions = data['%s_captions' % split][mask]
    image_idxs = data['%s_image_idxs' % split][mask]
    image_features = data['%s_features' % split][image_idxs]
    urls = data['%s_urls' % split][image_idxs]
    return captions, image_features, urls


def decode_captions(captions, idx_to_word):
    """
    Decode output captions into worded captions.
    Arguments:
        captions: output captions to be decoded
        idx_to_word: dictionary of word-index vocabulary table
    Outputs:
        decoded: decoded worded captions
    """
    singleton = False
    if captions.ndim == 1:
        singleton = True
        captions = captions[None]
    decoded = []
    N, T = captions.shape
    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t]]
            if word != '<NULL>':
                words.append(word)
            if word == '<END>':
                break
        decoded.append(' '.join(words))
    if singleton:
        decoded = decoded[0]
    return decoded


def image_from_url(url):
    """
    Read an image from a URL. Returns a numpy array with the pixel data.
    Arguments:
        url: urls for images for display
    Outputs:
        img: numpy array for the image
    """
    try:
        f = urllib.request.urlopen(url)
        _, fname = tempfile.mkstemp()
        with open(fname, 'wb') as ff:
            ff.write(f.read())
        img = imread(fname)
        # os.remove(fname)
        return img
    except urllib.error.HTTPError as e:
        print('HTTP Error: ', e.code, url)
    except urllib.error.URLError as e:
        print('URL Error: ', e.reason, url)
