import torch
from network.networkUtility import train
from Dataset import Dataset, train_valid_split
from network.networkUtility import prepareTraining
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
from network.networkUtility import BATCH_SIZE
from network.networkUtility import loadModelDeepForensics, saveModel
from network.networkUtility import randomSearchCoarse
from network.networkUtility import randomSearchFine
from network.networkUtility import loadHypeparameter

train_transform = transforms.Compose([transforms.Resize(333),
                                      transforms.CenterCrop(299),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

training_set = Dataset('/aiml/project/DFDC/FramesDataset_full/train', transform=train_transform)
type_optimizer = 'SGD' # 'Adam' / 'RMSprop' / 'Adamax'
# Prova di training
model = loadModelDeepForensics()
prepareTraining(model,type_optimizer)
train(model, train_dataloader, val_dataloader,type_optimizer)    # <-- mettere il validation set nell'ultimo argomento

'''# Prova di hyperparameters optimization
randomSearchCoarse(train_dataloader, train_dataloader,type_optimizer)
randomSearchFine(train_dataloader, train_dataloader,type_optimizer)

# Prova di visualizzazione dei risultati del hyperparameters optimization
avg_accuracy_list = []
LR_list = []
WEIGHT_DECAY_list = []
STEP_SIZE_list = []

path_init = '/aiml/project/utility/coarse/opt_hyper_coarse_'
for i in range(1):
  path = path_init + str(i)
  avg_acc, lr, weight_decay, step_size = loadHypeparameter(path)
  avg_accuracy_list.append(avg_acc); LR_list.append(lr); WEIGHT_DECAY_list.append(weight_decay); STEP_SIZE_list.append(step_size)
  print(avg_acc, lr, weight_decay, step_size)


# select the best hyperparameters
i_max = avg_accuracy_list.index(max(avg_accuracy_list))
LR = LR_list[i_max]
WEIGHT_DECAY_list = WEIGHT_DECAY_list[i_max]
STEP_SIZE = STEP_SIZE_list[i_max]
'''