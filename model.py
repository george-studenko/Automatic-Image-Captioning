import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F



class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size 
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first = True) #,
        self.fc = nn.Linear(hidden_size,vocab_size)                    
        self.hidden = self.init_hidden()
        
    def init_hidden(self):        
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))
    
    
    def forward(self, features, captions):
        captions = captions[:,:-1]
        embeds = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(1), embeds), 1)
        
        lstm_output, self.hidden_ = self.lstm(inputs)
        output = self.fc(lstm_output)
        
        return output
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        caption = []
        for idx in range(max_len):
            lstm_output, states = self.lstm(inputs,states)
            output = self.fc(lstm_output.squeeze(1))
            predicted_class = output.max(dim=1)[1]
            caption.append(predicted_class.item())
            inputs = self.embed(predicted_class).unsqueeze(1)
        return caption
