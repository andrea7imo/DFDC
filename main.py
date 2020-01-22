import time
import torch
from network.networkUtility import train
from Dataset import Dataset, train_valid_split
from network.networkUtility import prepareTraining
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
from network.networkUtility import BATCH_SIZE, NUM_ITER
from network.networkUtility import loadModelDeepForensics, saveModel
from network.networkUtility import randomSearchCoarse
from network.networkUtility import randomSearchFine
from network.networkUtility import loadHypeparameter

#%%
train_transform = transforms.Compose([transforms.Resize(333),
                                      transforms.CenterCrop(299),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

training_set = Dataset('/aiml/project/DFDC/FramesDataset_full/train', transform=train_transform, max_real=5000, max_fake=5000)

train_idx, valid_idx = train_valid_split(training_set, 2)

valid_dataset = Subset(training_set, valid_idx)
train_dataset = Subset(training_set, train_idx)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)

print('Train Dataset: {}'.format(len(train_dataset)))
print('Validation Dataset: {}'.format(len(valid_dataset)))

#%%
# Scelta ottimizzatore
type_optimizer = 'Adam' #/ 'SGD' / 'RMSprop' / 'Adamax'

# Prova di training e salvataggio
#transfer_model = loadModelDeepForensics()
#prepareTraining(transfer_model, type_optimizer)
#train(model, train_dataloader, valid_dataloader, type_optimizer)    # <-- mettere il validation set nell'ultimo argomento
#saveModel(None, transfer_model.model.state_dict(), None, None, None, '/home/leonardo/Scrivania/testing.pth')

# Prova di hyperparameters optimization
tic = time.perf_counter()
randomSearchCoarse(train_dataloader, valid_dataloader, type_optimizer)
toc = time.perf_counter()

#tic = time.perf_counter()
#randomSearchFine(train_dataloader, valid_dataloader, type_optimizer)
#toc = time.perf_counter()

elapsed_time = time.strftime('%H:%M:%S', time.gmtime(toc-tic))
print(f"Search time: {elapsed_time}")

#%%
# Prova di visualizzazione dei risultati del hyperparameters optimization
avg_accuracy_list = []
F1_list = []
LR_list = []
WEIGHT_DECAY_list = []
STEP_SIZE_list = []


path_init = '/aiml/project/DFDC/Outputs/coarse/opt_hyper_coarse_'
for i in range(NUM_ITER):
  path = path_init + str(i)
  avg_acc, f1, lr, weight_decay, step_size = loadHypeparameter(path)
  avg_accuracy_list.append(avg_acc); F1_list.append(f1); LR_list.append(lr); WEIGHT_DECAY_list.append(weight_decay); STEP_SIZE_list.append(step_size)
  print(avg_acc, f1, lr, weight_decay, step_size)

# select the best hyperparameters
i_max = avg_accuracy_list.index(max(avg_accuracy_list))
F1 = F1_list[i_max]
LR = LR_list[i_max]
WEIGHT_DECAY_list = WEIGHT_DECAY_list[i_max]
STEP_SIZE = STEP_SIZE_list[i_max]
