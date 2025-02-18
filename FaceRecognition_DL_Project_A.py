from cProfile import label
import pathlib
from pickle import TRUE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import os
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import math
import intel_extension_for_pytorch as ipex
import time
import pandas as pd
import csv
import cv2
from torchsummary import summary
import copy


batch_size = 128
margin=1.0
num_epochs = 30
lr = 0.0005

multi_family_dataset_train = 0
multi_family_dataset_test = 0
train_data_loader = 0
test_data_loader = 0
criterion = 0
optimizer = 0
exp_lr_scheduler = 0
device = 'cpu'


test_dataset_path='/home/gta/ksatya/recognizing-faces-in-the-wild/test/'
test_relation_csv_file_path='/home/gta/ksatya/recognizing-faces-in-the-wild/sample_submission.csv'
train_dataset_path='/home/gta/ksatya/recognizing-faces-in-the-wild/train'
train_data_relations_csv_path='/home/gta/ksatya/recognizing-faces-in-the-wild/train_relationships.csv'

# Transformations
train_transform = transforms.Compose([
    #transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class MultiFamilyDataset(Dataset):
    def __init__(self, root_dir, csv_file, batch_size, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = {}
        self.batch_size = batch_size
        self.relationships = []
        self.prev_family_member1 = 0
        self.prev_family_member2 = 0
        # Read the CSV file to get relationships
        self.relationships = pd.read_csv(csv_file, header=None).values.tolist()
        
        # Traverse the root directory to collect images for each family
        for family_folder in os.listdir(root_dir):
            family_path = os.path.join(root_dir, family_folder)
            if os.path.isdir(family_path):
                self.data[family_folder] = {}
                for member_folder in os.listdir(family_path):
                    member_path = os.path.join(family_path, member_folder)
                    if os.path.isdir(member_path):
                        images = []
                        for img_name in os.listdir(member_path):
                            if img_name.endswith('.jpg'):
                                img_path = os.path.join(member_path, img_name)
                                images.append(img_path)
                        if len(images) > 1:  # Ensure there are at least 2 images
                            self.data[family_folder][member_folder] = images

        # Calculate total number of pairs and round to nearest multiple of batch_size
        total_images = sum(len(images) for family in self.data.values() for images in family.values())
        self.total_pairs = math.floor(total_images / self.batch_size) * self.batch_size
        

    def __len__(self):
        return self.total_pairs

    def __getitem__(self, idx):
        # Randomly choose whether to return a positive or negative pair
        should_get_same_class = random.randint(0, 1)
        
        if should_get_same_class:
            # Choose a positive pair from the CSV file
            
            while True:
                try:
                    family_member1, family_member2 = random.choice(self.relationships)
                    img1 = random.choice(self.get_images(family_member1))
                    img2 = random.choice(self.get_images(family_member2))
                    break;
                except:
                    #print("Invalid folder detected. Trying for another relationship entry from csv file!!")
                    continue
            label = 1  # Positive pair
         
        else:
            # Choose a negative pair by selecting unrelated members
            family1, family2 = random.sample(list(self.data.keys()), 2)
            member1 = random.choice(list(self.data[family1].keys()))
            member2 = random.choice(list(self.data[family2].keys()))
            img1 = random.choice(self.data[family1][member1])
            img2 = random.choice(self.data[family2][member2])
            label = 0  # Negative pair
        
        img1 = Image.open(img1).convert('RGB')
        img2 = Image.open(img2).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, label

    def get_images(self, family_member):
        """
        Method to retrieve images for a specific family member.
        family_member: str, format 'FamilyFolder/MemberFolder'
        """
        family_folder, member_folder = family_member.split('/')
        return self.data[family_folder][member_folder]


def createDataSet():
    global multi_family_dataset_train
    global multi_family_dataset_test
    global train_data_loader
    global test_data_loader

    # Create the dataset and dataloader
    multi_family_dataset_train = MultiFamilyDataset(root_dir = train_dataset_path, 
                                          csv_file = train_data_relations_csv_path,
                                          batch_size = batch_size, 
                                          transform=train_transform)
    multi_family_dataset_test = MultiFamilyDataset(root_dir = train_dataset_path, 
                                          csv_file= train_data_relations_csv_path,
                                          batch_size = batch_size, 
                                          transform=test_transform)

    dataset_sizes = len(multi_family_dataset_train)
    print("There are {} training images".format(dataset_sizes))

    # Split the dataset into train and test sets
    train_data_loader = DataLoader(multi_family_dataset_train, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(multi_family_dataset_test, batch_size=batch_size, shuffle=False)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        self.resnet = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V2)
        
        for param in self.resnet.parameters():  
            param.requires_grad = False
               
        # Replace the fully connected layer (original classification layer)
        # Original resnet.fc is an nn.Linear layer; we replace it with a custom layer sequence
        num_ftrs = self.resnet.fc.in_features  # This is the number of input features to the original fc layer
        
        
        # We replace the final fully connected layer with a new one, then apply sigmoid
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1024),  # First fully connected layer
            nn.ReLU(),                  # Apply RELU activation
            #nn.Dropout(p=0.7),         # Apply dropout for regularization
            nn.Linear(1024, 256),       # Second fully connected layer to reduce dimension to 512
            nn.ReLU(),                  # Apply RELU activation
            #nn.Dropout(p=0.5),         # Apply dropout for regularization
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        output = self.resnet(x)
                       
        return output

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        #Euclidean distance between two vectors. Identifies how far and close are the vectors are.
        euclidean_distance = nn.functional.pairwise_distance(output1, output2, keepdim=True)
        #Find the loss of simillar pairs. label = 0 --> different, 1 --> Same
        pos = (1-label) * torch.pow(euclidean_distance, 2)
        #Find the loss of dis-simillar pairs. label = 0 --> different, 1 --> Same
        neg = label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        #Find the mean of loss.
        loss_contrastive = torch.mean( pos + neg )
        return euclidean_distance, loss_contrastive
    
def getDeviceForTrainAndTest(model):
    global device
    if torch.cuda.is_available():
        device = "cuda"
        model = model.cuda()
    elif torch.xpu.is_available():
        device = "xpu"
        #print(f"Using {torch.xpu.device_count()} GPUs")
        #model = model.DataParallel(model)
        model = model.to("xpu")
    else:
        device = "cpu"
        model = model.to("cpu")
    
    return model
###################################################################################
#Train the model
###################################################################################

def train_model(model):
    global criterion
    global optimizer
    global exp_lr_scheduler
    global num_epochs
    global multi_family_dataset_train
    global multi_family_dataset_test
    global train_data_loader
    global test_data_loader
    global device

    print(f"Model will be trained and tested on device: {device}")

    #Train the model
    print("Training the model: \n")
    since = time.time()

    #initialize model weights and accuracy
    best_model_wts = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    for epoch in range (num_epochs):
        #Train the model
        model.train()
        #Reset the correct to 0 after passing through all the dataset
        correct = 0        
        loss = 0
        train_loss = 0
        train_total = 0

        for image1, image2,labels in train_data_loader:
            image1 = Variable(image1)
            image2 = Variable(image2)
            labels = Variable(labels)
            #print(labels)
            if "cuda" == device:
                image1 = image1.cuda()
                image2 = image2.cuda()
                labels = labels.cuda()
            elif "xpu" == device or "cpu" == device:
                image1 = image1.to(device)
                image2 = image2.to(device)
                labels = labels.to(device)

            optimizer.zero_grad()
            output1 = model(image1)
            output2 = model(image2)
            distance, loss = criterion(output1, output2, labels)
            loss += loss.item()
            loss.backward()
            optimizer.step()
            exp_lr_scheduler.step()
            train_total += labels.size(0)

        train_loss = loss/len(train_data_loader)
        
        #Test the model and find accuracy.
        model.eval()
        correct = 0
        test_total = 0
        loss = 0
        for image1, image2,labels in test_data_loader:
            image1 = Variable(image1)
            image2 = Variable(image2)
            labels = Variable(labels)
            #print(labels)
            if "cuda" == device:
                image1 = image1.cuda()
                image2 = image2.cuda()
                labels = labels.cuda()
            elif "xpu" == device or "cpu" == device:
                image1 = image1.to(device)
                image2 = image2.to(device)
                labels = labels.to(device)

            output1 = model(image1)
            output2 = model(image2)
            distance, loss = criterion(output1, output2, labels)
            loss += loss.item()

            predicted = (distance < 0.5)
            correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

            if test_total > 0.2 * train_total:
                break;


        accuracy = correct/test_total
        print("Epoch [{}/{}] ----> Training loss : {:.4f} Test accuracy :{:.4f} \n".format(epoch+1,num_epochs,train_loss,accuracy))

        if accuracy > best_accuracy:
            print("Previous best accuracy:{:.4f}, current epochs accuracy:{:.4f}".format(best_accuracy, accuracy))
            best_accuracy = accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, 'FaceRecognition.pth')

    time_elapsed = time.time() - since
    print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

def test_csv_relationships(model, test_dataset_path, test_relation_csv_file_path):
    ###################################################################################
    #Test the model
    ######################################################################################
    global device
    inCsvFile = 0
    outCsvFile = 0
    parsedEntries = 0
    #Try to open the file if exists.
    try:
        inCsvFileHandle = open(test_relation_csv_file_path, mode ='r')
        inCsvFile = csv.reader(inCsvFileHandle)
        outCsvFileHandle = open(test_relation_csv_file_path+"_processed", mode ='w')
        outCsvFile = csv.writer(outCsvFileHandle)
    except:
        print("No test relation CSV file found. Exiting!!!")
        exit(0) 

    since = time.time()
    #Load from saved model
    print("Loading the model for testing..")
    model.load_state_dict(torch.load('FaceRecognition.pth'))
    # Test the model
    model.eval()
    print("Training the images as per the csv file")  
    with torch.no_grad():
        for rows in inCsvFile:
            #Skip first row.
            if 'img_pair' == rows[0]:
                outCsvFile.writerow(rows)
                continue
            out_rows=[]
            image_pair = rows[0]
            image1, image2 = image_pair.split("-")
            image1 = cv2.imread(test_dataset_path + image1, 1) #Take all 3 channels
            image1 = Image.fromarray(image1)
            image1 = test_transform(image1)
            image1 = image1.view(1,3,224,224)
            image2 = cv2.imread(test_dataset_path + image2, 1) #Take all 3 channels
            image2 = Image.fromarray(image2)
            image2 = test_transform(image2)
            image2 = image2.view(1,3,224,224)

            image1 = Variable(image1)
            image2 = Variable(image2)

            if "cuda" == device:
                image1 = image1.cuda()
                image2 = image2.cuda()

            elif "xpu" == device or "cpu" == device:
                image1 = image1.to(device)
                image2 = image2.to(device)


            output1 = model(image1)
            output2 = model(image2)
            
            #Calculate the distance.
            distance = nn.functional.pairwise_distance(output1, output2)
            
            #print("distance: {:.4f}".format(distance.item()))
            #distance = torch.sqrt(torch.sum(torch.pow(torch.subtract(output1, output2), 2), dim=0))  
            #predicted = (distance.item() < 0.5)
            #print("row:{}, distance:{} predicted:{}".format(rows[0], distance.mean(), predicted))
            #total += 1
            #if total > 5:
            #    break
            out_rows.append(rows[0])
            if distance.item() < 0.5:
                out_rows.append(1)
            else:
                out_rows.append(0)

            #out_rows.append(distance.item())

            #Write prediction to output row
            outCsvFile.writerow(out_rows)
            
            parsedEntries += 1
            if parsedEntries % 1000 == 0:
                print("Parsed {} entries".format(parsedEntries))

    time_elapsed = time.time() - since
    print(f'Testing completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')


def main():
    global criterion
    global optimizer
    global exp_lr_scheduler

    #Load the ResNet
    model = SiameseNetwork()
    
    model_file = pathlib.Path("FaceRecognition.pth")
    #Check whether pre-trained model exists or not.
    if model_file.is_file():       
        print("Pre-trained model exists. So, going for testing part")
        #Get the device to be used for training and move model to that device
        model = getDeviceForTrainAndTest(model)
        #Test relationships from CSV file.
        test_csv_relationships(model, test_dataset_path, test_relation_csv_file_path)
    else:
        print("Pre trained model does not exist. So, training model before testing")

        criterion = ContrastiveLoss(margin)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        #Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        #Show the model
        summary(model = model, input_size=[(3,224,224)] ,batch_size=batch_size)

        #Get the device to be used for training and move model to that device
        model = getDeviceForTrainAndTest(model)
        createDataSet()
        #Train the model
        train_model(model)
        #Test relationships from CSV file.
        test_csv_relationships(model, test_dataset_path, test_relation_csv_file_path)
  
if __name__ == '__main__':
    main()

        