import numpy as np
import matplotlib.pyplot as plt
import pickle
from CaptionRNN import RNNImageCaption
from train import CaptionTrain
from utils import *

### set seed
##np.random.seed(1)
### load dataset for model and dataset for training
data = load_coco_dataset(PCA_features=True)
subset_data = load_coco_dataset()#max_train=10000)
##np.save('save/subset_data.npy', subset_data)  # save dataset
### create model instance
##sub_rnn_model = RNNImageCaption(cell_type='rnn', word_to_idx=data['word_to_idx'],
##          input_dim=data['train_features'].shape[1], hidden_dim=512, wordvec_dim=256,)
### train model
##sub_rnn_solver = CaptionTrain(subset_data, sub_rnn_model, update='adam', num_epochs=50,
##                                batch_size=100, update_params={'lr': 5e-3}, lr_decay=0.95, print_freq=100)
##sub_rnn_solver.train()

# plot the training losses
# plt.plot(sub_rnn_solver.loss_history)
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.title('Training loss history')
# plt.show()

# save model file
##with open('save/sub_rnn_model.pickle', 'wb') as f:
##    pickle.dump(sub_rnn_model, f)
## restore model file
with open('save/sub_rnn_model.pickle', 'rb') as f:
    sub_rnn_model = pickle.load(f)

## Image representation for the model performance
for split in ['train', 'val']:
    # sample mini-batches and ground truth captions
    minibatch = sample_coco_minibatch(subset_data, split=split, batch_size=3)
    gt_captions, features, urls = minibatch
    print(minibatch)
    print(features)
    gt_captions = decode_captions(gt_captions, data['idx_to_word'])
    # generate captions from model
    sample_captions = sub_rnn_model.generate_captions(features)
    print(sample_captions)
    sample_captions = decode_captions(sample_captions, data['idx_to_word'])
    # show images with captions
    for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
        plt.imshow(image_from_url(url))
        plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
        plt.axis('off')
        plt.show()
