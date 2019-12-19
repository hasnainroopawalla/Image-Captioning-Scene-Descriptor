import numpy as np
from builtins import object
from layers import *


class RNNImageCaption(object):
    """
    Define a RNN_image_captioning class, the instance of which outputs captions given image features.
    """
    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128, hidden_dim=128, cell_type='rnn', dtype=np.float32):
        """
        Initialization of instance in RNN_image_captioning.
        Arguments:
             word_to_idx: dictionary of word-index vocabulary table with V entries
             input_dim: input image feature dimension D
             wordvec_dim: word vector dimension W
             hidden_dim: hidden state dimension H in RNN
             cell_type: either 'rnn' or 'lstm' setting the RNN type
             dtype: numpy datatype - float32 for training and float64 for numerical gradient check
        """
        if cell_type not in ['rnn', 'lstm']:
            raise ValueError('Unknown cell type of "%s"' % cell_type)
        self.cell_type = cell_type
        self.input_dim = input_dim
        self.wordvec_dim = wordvec_dim
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.params = {}
        # save indices of NULL, START and END
        self.null = word_to_idx['<NULL>']
        self.start = word_to_idx.get('<START>')
        self.end = word_to_idx.get('<END>')
        # initialization of word vectors
        self.params['W_embed'] = np.random.randn(len(word_to_idx), wordvec_dim) / 100
        # initialization of hidden state projection parameters for CNN
        self.params['W_proj'] = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.params['b_proj'] = np.zeros(hidden_dim)
        # initialization of RNN parameters
        dimension_factor = {'rnn':1, 'lstm':4}[cell_type]
        self.params['Wx'] = np.random.randn(wordvec_dim, dimension_factor * hidden_dim) / np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dimension_factor * hidden_dim) / np.sqrt(hidden_dim)
        self.params['b'] = np.zeros(dimension_factor * hidden_dim)
        # initialization of vocab weights
        self.params['W_vocab'] = np.random.randn(hidden_dim, len(word_to_idx)) / np.sqrt(hidden_dim)
        self.params['b_vocab'] = np.zeros(len(word_to_idx))
        # cast dtype
        for para_name, param in self.params.items():
            self.params[para_name] = param.astype(self.dtype)

    def loss(self, features, captions):
        """
        Calculate the training loss for captioning RNN.
        Arguments:
             features: input image features with shape of (N, D)
             captions: an integer array of ground-truth captions with shape of (N, T) with elements in [0, V)
        Outputs:
            loss: float of loss value
            grads: dictionary of gradients of parameters in self.params
        """
        # Cut out the last words of captions as input, and the expected output is everything but the first words.
        # Note that the first element of captions would be the START token.
        captions_in = captions[:, :-1]  # entire caption except for the last words
        captions_out = captions[:, 1:]  # entire caption except for the first words

        mask = (captions_out != self.null)  # Indicating non-NULL indices to be used

        # unpack initialized parameters
        W_embed = self.params['W_embed']
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        # loss calculation
        h0 = np.dot(features, W_proj) + b_proj  # initial hidden state from image features - (N, H)
        x, cache_embed = word_embedding_forward(captions_in, W_embed)  # transform words in captions_in - (N, T, W)
        if self.cell_type == 'rnn':  # use Vanilla RNN to produce hidden states from input word vectors - (N, T, H)
            h, cache_rnn = rnn_forward(x, h0, Wx, Wh, b)
        else:                        # use LSTM to produce hidden states from input word vectors - (N, T, H)
            h, cache_lstm = lstm_forward(x, h0, Wx, Wh, b)
        scores, cache_temporal = temporal_affine_forward(h, W_vocab, b_vocab)  # compute scores - (N, T, V)
        loss, dscores = temporal_softmax_loss(scores, captions_out, mask)  # compute loss, ignoring <NULL> tokens

        # gradients calculation using back-props
        dh, dW_vocab, db_vocab = temporal_affine_backward(dscores, cache_temporal)
        if self.cell_type == 'rnn':
            dx, dh0, dWx, dWh, db = rnn_backward(dh, cache_rnn)
        else:
            dx, dh0, dWx, dWh, db = lstm_backward(dh, cache_lstm)
        dW_embed = word_embedding_backward(dx, cache_embed)
        dW_proj = np.dot(features.T, dh0)
        db_proj = np.sum(dh0, axis=0)

        # put gradients into dictionary
        # note that the keys have the same strings as in parameters for convenience during extraction
        grads = {}
        grads['W_embed'] = dW_embed
        grads['W_proj'], grads['b_proj'] = dW_proj, db_proj
        grads['Wx'], grads['Wh'], grads['b'] = dWx, dWh, db
        grads['W_vocab'], grads['b_vocab'] = dW_vocab, db_vocab

        return loss, grads

    def generate_captions(self, features, max_length=100):
        """
        Generate captions from the image features.
        Arguments:
             features: input image features with shape of (N, D)
             max_length: maximum length T of generated caption
        Outputs:
            captions: array of generated captions with shape of (N, T) and each element lies in [0, V)
        """
        
        N, D = features.shape
        captions = self.null * np.ones((N, max_length), dtype=np.int32)  # initialize captions to <NULL>s

        # Unpack parameters
        W_embed = self.params['W_embed']
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        # Generate captions
        h0 = np.dot(features, W_proj) + b_proj  # initial hidden state from image features - (N, H)
        captions[:, 0] = self.start  # set <START> tokens to the generated captions
        capt = self.start * np.ones((N, 1), dtype=np.int32)  # set <START> tokens to the generated word for each time step
        prev_h = h0
        prev_c = np.zeros(h0.shape)  # initialize the cell state to zeros
        for t in range(max_length):
            x, _ = word_embedding_forward(capt, W_embed)  # word embedding
            # get next hidden state
            if self.cell_type == 'rnn':
                h, _ = rnn_step_forward(np.squeeze(x), prev_h, Wx, Wh, b)  # note: squeeze for dimension match
                prev_h = h
            else:
                h, c, _ = lstm_step_forward(np.squeeze(x), prev_h, prev_c, Wx, Wh, b)
                prev_h = h
                prev_c = c
            scores, _ = temporal_affine_forward(h[:, np.newaxis, :], W_vocab, b_vocab)  # note: new axis for dimension match
            capt = np.squeeze(np.argmax(scores, axis=2))
            captions[:, t] = capt  # store generated captions
        print(captions)
        return captions
