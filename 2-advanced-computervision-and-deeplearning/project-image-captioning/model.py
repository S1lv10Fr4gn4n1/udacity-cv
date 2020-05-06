import torch
import torch.nn as nn
import torchvision.models as models

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
        # define the embedding layer for the captions
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # define the LSTM layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        # define the linear layer
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        # drop the last caption
        embed = self.embedding(captions[:, :-1])
        # contact features and embedding captions
        embed = torch.cat((features.unsqueeze(1), embed), dim = 1)
        # pass to lstm the sequence of inputs
        lstm_outputs, _ = self.lstm(embed)
        # pass to linear layer the lstm output and return
        return self.fc(lstm_outputs)
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sentence = []
        for i in range(max_len):
            lstm_outputs, states = self.lstm(inputs, states)
            lstm_outputs = lstm_outputs.squeeze(1)
            out = self.fc(lstm_outputs)
            last_pick = out.max(1)[1]
            sentence.append(last_pick.item())
            inputs = self.embedding(last_pick).unsqueeze(1)
        
        return sentence