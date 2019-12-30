import torch
from network.networkUtility import train
from Dataset import Dataset
from network.networkUtility import prepareTraining
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
from network.networkUtility import BATCH_SIZE

model = torch.load('/aiml/references/faceforensics++_models_subset/full/xception/full_c23.p')
model = model.cuda()

train_transform = transforms.Compose([transforms.Resize(333),
                                      transforms.CenterCrop(299),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

training_set = Dataset('/aiml/project/DFDC/FramesDataset_full/train', transform=train_transform)
train_dataloader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)
# introdurre il validation set

prepareTraining(model)
train(model, train_dataloader, train_dataloader) # <-- mettere il validation set nell'ultimo argomento