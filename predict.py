import numpy as np
import argparse
import json
import utility_functions

ap = argparse.ArgumentParser(description='predict.py')

ap.add_argument('input', default = "flowers/test/1/image_06743.jpg")
ap.add_argument('checkpoint', default = "checkpoint.pth")
ap.add_argument('--top_k', default = 5, dest = "top_k", action = "store", type = int)
ap.add_argument('--category_names', dest = "category_names", action = "store", default = 'cat_to_name.json')
ap.add_argument('--gpu', default = False, action = 'store_true') 

pa = ap.parse_args()
image_path = pa.input
path = pa.checkpoint
number_of_outputs = pa.top_k
category_names = pa.category_names
power = pa.gpu

print("Loading datasets...")
train_loaders, valid_loaders, test_loaders = utility_functions.load_data()

print("Loading checkpoint...")
model = utility_functions.load_checkpoint(path)
utility_functions.load_data(data_dir)

print("Running the Program. This may take a while...")
probabilities = Util.predict(image_path, model, 5, hardware)


print("Here are the results...")
with open(category_names, 'r') as json_file:
    cat_to_name = json.load(json_file)

labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
probability = np.array(probabilities[0][0])

i=0
while i < number_of_outputs:
    print("{} with a probability of {}%".format(labels[i], probability[i]*100))
    
print("...END OF SESSION...")