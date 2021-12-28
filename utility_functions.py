import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image


def load_data(data_dir="./flowers" ):

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Defining transforms for the training, validation, and testing sets
    orig_img_size = 256
    new_img_size = 224
    means = [0.485, 0.456, 0.406]
    std_dev = [0.229, 0.224, 0.225]
    
    train_transform_set = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(new_img_size),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(means, std_dev)])

    test_transform_set = transforms.Compose([transforms.Resize(orig_img_size),
                                          transforms.CenterCrop(new_img_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize(means, std_dev)])

    valid_transform_set = transforms.Compose([transforms.Resize(orig_img_size),
                                                transforms.CenterCrop(new_img_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize(means, std_dev)])
        
    #Loading the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform = train_transform_set)
    valid_dataset = datasets.ImageFolder(valid_dir, transform = valid_transform_set)
    test_dataset  = datasets.ImageFolder(test_dir , transform = test_transform_set)

    #Using the image datasets and the trainforms, define the dataloaders
    train_loaders = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loaders = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)
    test_loaders  = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    return train_loaders , valid_loaders, test_loaders


def model_setup(architecture ='vgg16', dropout=0.5, hidden_layers=512, learning_rate=0.01, power='gpu'):

    architectures = { "vgg16":25088,
                    "inception":1024,
                    "alexnet":9216 }

    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif architecture == 'inception':
        model = models.inception(pretrained=True)
    elif architecture == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("Sorry but {} is not a valid architecture. Please try vgg16, inception or alexnet.".format(architecture))

    # Freeze parameters so we don't backprop through them
    for param in model.parameters(): 
        param.requires_grad = False

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                                ('dropout',nn.Dropout(dropout)),
                                ('inputs', nn.Linear(architectures[architecture], hidden_layers)),
                                ('relu1', nn.ReLU()),
                                ('hidden_layer_1', nn.Linear(hidden_layers, 80)),
                                ('relu2',nn.ReLU()),
                                ('hidden_layer_2',nn.Linear(80, 70)),
                                ('relu3',nn.ReLU()),
                                ('hidden_layer_3',nn.Linear(70, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                            ]))

    model.classifier = classifier
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    if torch.cuda.is_available() and power == 'gpu':
        model.cuda()

    return model, optimizer, criterion

    
def train(train_loaders, valid_loaders, model, optimizer, criterion, epochs = 20, print_every = 5, power = 'gpu'):
    steps = 0
    
    print("--------------Training has started------------- ")  
    train_losses, valid_losses = [], []
    for e in range(epochs):
        running_loss = 0

        for ii, (inputs, labels) in enumerate(train_loaders):
            steps += 1

            # Move input and label tensors to the GPU (if available)
                            # Move input and label tensors to the GPU (if available)
            if torch.cuda.is_available() and power == 'gpu':
                train_inputs, train_labels = train_inputs.to('cuda'), train_labels.to('cuda')
                inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Deactivate dropout
                model.eval()
                valid_loss = 0
                accuracy = 0

                for ii, (valid_inputs, valid_labels) in enumerate(valid_loaders):
                    optimizer.zero_grad()

                    # Move input and label tensors to the GPU (if available)
                    if torch.cuda.is_available() and power == 'gpu':
                        valid_inputs, valid_labels = valid_inputs.to('cuda'), valid_labels.to('cuda')
                        model.to('cuda')

                    with torch.no_grad():    
                         outputs = model.forward(valid_inputs)
                         valid_loss = criterion(outputs, valid_labels)
                         ps = torch.exp(outputs).data 
                         equality = (valid_labels.data == ps.max(1)[1])
                         accuracy += equality.type_as(torch.FloatTensor()).mean()

                valid_loss = valid_loss / len(valid_loaders)
                train_loss = running_loss / len(train_loaders)

                train_losses.append(train_loss)
                valid_losses.append(valid_loss)

                accuracy = (accuracy /len(valid_loaders)) * 100

                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss {:.4f}".format(valid_loss),
                      "Accuracy: {:.4f}".format(accuracy))

                running_loss = 0
                
                
    print("--------------Training has ended--------- ")
    print("----------Number of steps {}--------------".format(steps))


def accuracy(model, test_loaders, power="gpu"):
    #Check the accurecy on test dataset
    accuracy = 0

    for ii, (inputs, labels) in enumerate(test_loaders):
        # Move input and label tensors to the GPU (if available)
        if torch.cuda.is_available() and power == 'gpu':
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

        ps = torch.exp(model(inputs))

        top_p, top_class = ps.topk(1, dim=1)

        equals = top_class == test_labels.view(*top_class.shape)

        accuracy += torch.mean(equals.type(torch.FloatTensor))

    accuracy = accuracy / len(test_loaders)
    print(f'The accuracy of the network on test images is: {accuracy.item()*100}%')                                                    
                                                    

def save_checkpoint(model, class_to_idx, path='checkpoint.pth', architecture='inception', hidden_layers=512, dropout=0.5, learning_rate=0.01, epochs=20):
    model.class_to_idx = _class_to_idx
    torch.save({
                'architecture': architecture,
                'hidden_layers': hidden_layers,
                'dropout': dropout,
                'learning_rate': learning_rate,
                'nb_of_epochs': epochs,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx},
                'checkpoint.pth')


def load_checkpoint(path='checkpoint.pth'):
    checkpoint = torch.load(path)

    architecture = checkpoint['architecture']
    hidden_layers = checkpoint['hidden_layers']
    dropout = checkpoint['dropout']
    learning_rate=checkpoint['learning_rate']

    model,_,_ = Util.model_setup(architecture, hidden_layers, dropout, learning_rate)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model


def process_image(image_path):

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(Image.open(image_path))

  
def predict(image_path, model, topk=5, power='gpu'):
    
    if torch.cuda.is_available() and power == 'gpu':
        model.to('cuda:0')

        img_torch = rocess_image(image_path).unsqueeze_(0).float()

        if power == 'gpu':
            with torch.no_grad():
                output = model.forward(img_torch.cuda())
        else:
            with torch.no_grad():
                output=model.forward(img_torch)

        probability = F.softmax(output.data,dim=1)

        return probability.topk(topk)