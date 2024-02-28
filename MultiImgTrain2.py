import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from dataset import get_train_val_sampler
from imbalanced import ImbalancedDatasetSampler
from sklearn.metrics import classification_report, confusion_matrix

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train_path',type=str)
parser.add_argument('--test_path',type=str)
parser.add_argument('--lr',type=float,default=1e-4)
parser.add_argument('--N_images',type=int,default=6)
parser.add_argument('--collapse_labels',type=int,default=None)
parser.add_argument('--bs_train',type=int,default=64)
parser.add_argument('--bs_test',type=int,default=128)
parser.add_argument('--model',type=str,default="resnet18")
parser.add_argument('--gpu','--list', nargs='+', default=None)
parser.add_argument('--binary', action="store_true")
parser.add_argument('--debug', action="store_true")
parser.add_argument('--balanced', type=int, default=0)
parser.add_argument('--epochs',type=int,default=1000)
parser.add_argument('--cutpaste', action="store_true")
args = parser.parse_args()

if "KW51" in args.train_path:
    from dataset import MultiImgDataset as MultiImgDataset2new
else:
    from dataset import MultiImgDataset2new
transformations = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
num_workers_train = 0 if args.debug else max(1,int(args.bs_train/8)) 
num_workers_test = 0 if args.debug else 8
##CODEFOR KW51
#train_dataset = MultiImgDataset(args.train_path, transform = transformations,N=args.N_images)
#test_dataset = MultiImgDataset(args.test_path, transform = transformations,N=args.N_images)

if args.test_path is not None:
    train_dataset = MultiImgDataset2new(args.train_path, transform = transformations,N=args.N_images,binary=args.binary,balanced=args.balanced,collapse_labels=args.collapse_labels,cutpaste=args.cutpaste)
    test_dataset = MultiImgDataset2new(args.test_path, transform = transformations,N=args.N_images,binary=args.binary,balanced=args.balanced,collapse_labels=args.collapse_labels,cutpaste=False)
    #test_dataset_grouped = MultiImgDataset2new(args.test_path, transform = transformations,N=args.N_images,binary=args.binary,balanced=args.balanced,group=True)
    train_size = len(train_dataset)
    test_size = len(test_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.bs_train,shuffle=True,num_workers=num_workers_train)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size = args.bs_test, shuffle=False,num_workers=num_workers_test) #max(1,int(args.bs_test/8))
    #test_loader_grouped= torch.utils.data.DataLoader(test_dataset_grouped,batch_size = int(args.bs_test/40), shuffle=False,num_workers=8) #int(args.bs_test/40)
    num_labels = 2 if args.binary else test_dataset.num_labels  #PUT IN THE NUMBER OF LABELS IN YOUR DATA
else:
    full_dataset = MultiImgDataset2new(args.train_path, transform = transformations,N=args.N_images,binary=args.binary,balanced=args.balanced,collapse_labels=args.collapse_labels)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    num_labels = 2 if args.binary else full_dataset.num_labels  #PUT IN THE NUMBER OF LABELS IN YOUR DATA
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.bs_train,shuffle=False,num_workers=num_workers_train)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size = args.bs_test, shuffle=False,num_workers=num_workers_test) #max(1,int(args.bs_test/8))
print("Number of labels set to: "+str(train_dataset.num_labels))

if args.model == 'densenet':
    model = models.densenet161(pretrained=True)
    classifier_input = model.classifier.in_features * args.N_images
    model.classifier = nn.Identity()
if args.model == 'resnet18':
    model = models.resnet18(pretrained=True)
    classifier_input = model.fc.in_features * args.N_images
    model.fc = nn.Identity()
if args.model == 'resnet152':
    model = models.resnet152(pretrained=True)
    classifier_input = model.fc.in_features * args.N_images
    model.fc = nn.Identity()
if args.model =='convnext':
    model = models.convnext_large(pretrained=True)
    classifier_input = model.classifier[2].in_features * args.N_images
    model.classifier = nn.Identity()
if args.model =='vit':
    model = models.vit_l_32(pretrained=True)
    classifier_input = model.heads[0].in_features * args.N_images
    model.heads = nn.Identity()
if args.model == 'txt':
    model = nn.Sequential (
         nn.Conv1d(1, 32, kernel_size=4,stride=2),
         nn.ReLU(inplace=True),
         nn.Conv1d(32, 64, kernel_size=4,stride=2),
         nn.ReLU(inplace=True),
         nn.Conv1d(64, 128, kernel_size=4,stride=2),
         nn.ReLU(inplace=True),
         nn.Conv1d(128, 256, kernel_size=4,stride=2),
         nn.ReLU(inplace=True))
    classifier_input = 12288 * args.N_images #model.out_features * args.N_images 

classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=classifier_input, out_features=1024),
    nn.ReLU(inplace=True),
    nn.Linear(in_features=1024, out_features=1024),
    nn.ReLU(inplace=True),
    #nn.Dropout(0.5),
    nn.Linear(in_features=1024, out_features=num_labels),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.gpu is not None:
    print("Let's use", str(args.gpu), "GPUs!")
    if (type(args.gpu) is list) and (len(args.gpu)>1):
        model = nn.DataParallel(model,device_ids=[int(val) for val in args.gpu])

model.to(device)
classifier.to(device)
if args.debug:
    pass
else:
    try:
        model = torch.compile(model)
        classifier = torch.compile(classifier)
    except:
        print("pytorch 2.0 required")

def train_model(model,loader):
    model.train()
    # Training the model
    counter = 0
    train_loss = 0
    for inputs, labels in loader:
        # Move to device
        inputs = inputs.to(device, dtype=torch.float) 
        num_frames = inputs.shape[4]
        labels = labels.to(device)
        # Clear optimizers
        optimizer.zero_grad()
        # Forward pass
        output_list = []
        for i in range(num_frames):
            input = inputs[...,i]
            output = model.forward(input)
            output_list.append(output)
        output_list = torch.cat(output_list,dim=1)
        output = classifier.forward(output_list)
        # Loss
        loss = criterion(output, labels)
        # Calculate gradients (backpropogation)
        loss.backward()
        # Adjust parameters based on gradients
        optimizer.step()
        # Add the loss to the training set's rnning loss
        train_loss += loss.item() #*inputs[0].size(0)
        
        # Print the progress of our training
        counter += 1
        train_loss += loss.item()
    train_loss = train_loss / (len(loader))
    print('Training Loss: {:.6f}'.format( train_loss))

def test_model(model,loader):
    # Evaluating the model
    model.eval()
    val_loss = 0
    accuracy = 0
    test_counter = 0
    running_corrects = 0
    total_test_samples = 0
    preds_list = []
    labels_list = []
    # Tell torch not to calculate gradients
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, dtype=torch.float) 
            num_frames = inputs.shape[4]
            if inputs.dim() < 6:
                inputs = inputs[...,None]
            num_runs = inputs.shape[5]

            labels = labels.to(device)

            output_super_list = []
            for j in range(num_runs):
                interm_inputs = inputs[...,j]
                output_list = []
                for i in range(num_frames):
                    input = interm_inputs[...,i]
                    output = model.forward(input)
                    output_list.append(output)
                output_list = torch.cat(output_list,dim=1)
                output = classifier.forward(output_list)
                output_super_list.append(output.detach())
                # Calculate Loss
                valloss = criterion(output, labels)
                # Add loss to the validation set's running loss
                val_loss += valloss.item() #*inputs[0].size(0)
                del output, input
            
            # Since our model outputs a LogSoftmax, find the real 
            # percentages by reversing the log function
            final_output = torch.mean(torch.stack(output_super_list), dim=0)
            _, preds = torch.max(final_output, 1)
            running_corrects += torch.sum(preds == labels.data).item()
            total_test_samples += inputs.shape[0]

            ##save preds and labels for futher testing
            preds_list += preds.tolist()
            labels_list += labels.data.tolist()
    # Get the average loss for the entire epoch
    valid_loss = val_loss/(len(loader))
    # Print out the information
    epoch_acc = running_corrects / total_test_samples
    print('Accuracy: ', epoch_acc)
    print('Epoch: {} \tValidation Loss: {:.6f}'.format(epoch, valid_loss))
    diag_matrix = confusion_matrix(labels_list, preds_list,normalize="true").diagonal()
    print('Per-class accuracy:')
    print(diag_matrix)
    print('Precision/recall:')
    print(classification_report(labels_list,preds_list,digits=3))


#criterion = nn.CrossEntropyLoss(weight = torch.FloatTensor(class_weights)).to(device) if args.binary else nn.CrossEntropyLoss() 
criterion = nn.CrossEntropyLoss()
optimizer = optim.RAdam(list(model.parameters())+list(classifier.parameters()),lr=args.lr)

torch.backends.cudnn.benchmark = True
for epoch in range(args.epochs):
    train_model(model,train_loader)
    torch.save(model.state_dict(),"model.pt")
    if epoch % 1 == 0:
        torch.cuda.empty_cache()
        test_model(model,test_loader)
#        torch.cuda.empty_cache()
#        print("Testing on 40 grouped runs:")
#        test_model(model,test_loader_grouped)
#        torch.cuda.empty_cache()
