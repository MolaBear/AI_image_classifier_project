import argparse
import utility_functions

ap = argparse.ArgumentParser(description = 'train.py')

ap.add_argument('data_dir', default = "flowers/")
ap.add_argument('--epochs', dest = "epochs", action="store", type=int, default=10)
ap.add_argument('--arch', dest = "arch", action="store", default="vgg16", type = str)
ap.add_argument('--hidden_layers', type = int, dest = "hidden_layers", action = "store", default = 512)
ap.add_argument('--gpu', default = False, action = 'store_true')
ap.add_argument('--save_dir', dest = "save_dir", action = "store", default = "./checkpoint.pth")
ap.add_argument('--learning_rate', dest = "learning_rate", action = "store", default = 0.01)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)

pa = ap.parse_args()

data_dir = pa.data_dir
path = pa.save_dir
learning_rate = pa.learning_rate
architecture = pa.arch
dropout = pa.dropout
hidden_layers = pa.hidden_layers
power = "gpu" if pa.gpu else "cpu"
epochs = pa.epochs
print_every = 10

print("Loading datasets...")
train_loaders, valid_loaders, test_loaders = utility_functions.load_data(data_dir)

print("Setting up model structure...")
model, optimizer, criterion = utility_functions.model_setup(architecture, dropout, hidden_layers, learning_rate, power)

print("Model Training...")
utility_functions.train(train_loaders, valid_loaders, model, optimizer, criterion, epochs, print_every, power)

print("Checking testing accuracy...")
utility_functions.accuracy(model, test_loaders, power)

print("Saving model to disk...")
utility_functions.save_checkpoint(model, train_dataset.class_to_idx, path, architecture, hidden_layers, dropout, learning_rate)

print("Done!")