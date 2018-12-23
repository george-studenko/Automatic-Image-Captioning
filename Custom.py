import sys
sys.path.append('/home/george/cocoapi/PythonAPI')

import os

import torch
from torchvision import transforms, datasets

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from model import EncoderCNN, DecoderRNN
from vocabulary import Vocabulary

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)

    transformations = transforms.Compose([
        transforms.Resize(225),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    img = transformations(img)

    np_image = np.array(img)

    return np_image


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing

    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.axis('off')
    if title != None:
        ax.set_title(title)
    ax.imshow(image);

    return ax


vocab_threshold=5
vocab_file='./vocab.pkl'
start_word="<start>"
end_word="<end>"
unk_word="<unk>"
annotations_file = os.path.join('/home/george/', 'cocoapi/annotations/image_info_test2014.json')

vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
            end_word, unk_word, annotations_file, True)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder_file = 'encoder-2.pkl'
decoder_file = 'decoder-2.pkl'

embed_size = 256
hidden_size = 512

vocab_size = len(vocab)

encoder = EncoderCNN(embed_size)
encoder.eval()
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
decoder.eval()

encoder.load_state_dict(torch.load(os.path.join('./models', encoder_file)))
decoder.load_state_dict(torch.load(os.path.join('./models', decoder_file)))

encoder.to(device)
decoder.to(device)



def clean_sentence(output):
    sentence = ''
    for idx in output:
        if idx == 0:
            continue
        if idx == 1:
            break;
        else:
            sentence += vocab.idx2word[idx] + ' '
    return sentence


def get_prediction(paths):
    for path in paths:
        image = process_image(path)
        #orig_image = image.copy()
        imshow(image)
        image = torch.from_numpy(image)
        image = image.to(device)

        image.unsqueeze_(0)


        image = image.to(device)
        features = encoder(image).unsqueeze(1)
        output = decoder.sample(features)
        sentence = clean_sentence(output)
        plt.title(sentence)
        plt.show()

#        print(sentence)


paths = ['custom_images/people/1.jpg',
         'custom_images/people/1.jpg',
         'custom_images/people/1.jpg',
         'custom_images/people/1.jpg',
         'custom_images/people/1.jpg',
         'custom_images/people/1.jpg',
         'custom_images/people/1.jpg',
         'custom_images/people/1.jpg',
         'custom_images/people/1.jpg',
         'custom_images/people/1.jpg']
get_prediction(paths)

