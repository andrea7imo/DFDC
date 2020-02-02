import time
import torch
import os
from network.networkUtility import train
from Dataset import Dataset, train_valid_split
from network.networkUtility import prepareTraining
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
from network.networkUtility import BATCH_SIZE, NUM_ITER, NUM_EPOCHS
from network.networkUtility import loadModelDeepForensics, saveModel, loadModel
from network.networkUtility import randomSearchCoarse
from network.networkUtility import randomSearchFine
from network.networkUtility import loadHypeparameter, setHyperparameter
from network.networkUtility import plotAccuracyAndLoss

#%%
train_transform = transforms.Compose([transforms.Resize(333),
                                      transforms.CenterCrop(299),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                      #transforms.Lambda(lambda x: x + (0.1**0.5)*torch.randn(3, 299, 299))
])

training_set = Dataset('/aiml/project/DFDC/FramesDataset_full/train', transform=train_transform, max_real=None, max_fake=None, over=True)

train_idx, valid_idx = train_valid_split(training_set, num_targets=2, train_size=0.7)

valid_dataset = Subset(training_set, valid_idx)
train_dataset = Subset(training_set, train_idx)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, drop_last=True)

print('Train Dataset: {}'.format(len(train_dataset)))
print('Validation Dataset: {}'.format(len(valid_dataset)))

#%%
# Scelta ottimizzatore
type_optimizer = 'Adam'  # [RMSprop/SGD/Adam/Adamax]
path_init = f'/aiml/project/DFDC/Outputs/fine/{type_optimizer}/opt_hyper_fine_'

#%%
# Training e salvataggio
transfer_model = loadModelDeepForensics()
prepareTraining(transfer_model, type_optimizer)

tic = time.clock_gettime(time.CLOCK_MONOTONIC)
best_epoch, best_model_wts, bestAccuracy, bestF_1, accuracies, accuraciesTrain, F_1s, loss_values = \
    train(transfer_model, train_dataloader, valid_dataloader, type_optimizer)
toc = time.clock_gettime(time.CLOCK_MONOTONIC)

saveModel(best_epoch, best_model_wts, loss_values, accuracies, accuraciesTrain, F_1s,
          f'/aiml/project/DFDC/Outputs/models/model_over-all-{NUM_EPOCHS}-fine_adam_best.pth')
                        # model name format: model_<max_real>-<max_fake>-<NUM_EPOCHS>-<hyp_id>[-<trans>].pth

elapsed_time = time.strftime('%H:%M:%S', time.gmtime(toc-tic))
print(f"Training time: {elapsed_time}")

#%%
# Hyperparameters coarse optimization
tic = time.clock_gettime(time.CLOCK_MONOTONIC)
randomSearchCoarse(train_dataloader, valid_dataloader, type_optimizer, path_init)
toc = time.clock_gettime(time.CLOCK_MONOTONIC)
elapsed_time = time.strftime('%H:%M:%S', time.gmtime(toc-tic))
print(f"Search time: {elapsed_time}")

#%%
# Hyperparameters fine optimization
tic = time.clock_gettime(time.CLOCK_MONOTONIC)
randomSearchFine(train_dataloader, valid_dataloader, type_optimizer, path_init)
toc = time.clock_gettime(time.CLOCK_MONOTONIC)
elapsed_time = time.strftime('%H:%M:%S', time.gmtime(toc-tic))
print(f"Search time: {elapsed_time}")

#%%
# Visualizzazione dei risultati del hyperparameters optimization
avg_accuracy_list = []
F1_list = []
avg_acc_f1_list = []
LR_list = []
WEIGHT_DECAY_list = []
STEP_SIZE_list = []

print("avg_acc f1     lr     wd      step_size")
for i in range(NUM_ITER):
  path = path_init + str(i)
  if not os.path.exists(path):
      print("WARNING: the number of files is lower than NUM_ITER!")
      break
  avg_acc, f1, lr, weight_decay, step_size = loadHypeparameter(path)
  avg_accuracy_list.append(avg_acc); F1_list.append(f1); LR_list.append(lr); WEIGHT_DECAY_list.append(weight_decay); STEP_SIZE_list.append(step_size)
  avg_acc_f1_list.append((avg_acc + f1)/2)
  print(f"{avg_acc:5.4f}  {f1:5.4f} {lr:5.4f} {weight_decay:5.5f} {step_size}")

#%%
# select the best hyperparameters
i_max = avg_acc_f1_list.index(max(avg_acc_f1_list))
ACC = avg_accuracy_list[i_max]
F1 = F1_list[i_max]
LR = LR_list[i_max]
WEIGHT_DECAY = WEIGHT_DECAY_list[i_max]
STEP_SIZE = STEP_SIZE_list[i_max]
print("\nBest result:")
print("avg_acc f1     lr     wd      step_size")
print(f"{ACC:5.4f}  {F1:5.4f} {LR:5.4f} {WEIGHT_DECAY:5.5f} {STEP_SIZE}")
setHyperparameter(LR, WEIGHT_DECAY, STEP_SIZE)

#%%
# carica i parametri e stampa i grafici
net, best_epoch, loss_values, accuracies, accuraciesTrain, f1s, best_model_wts = \
    loadModel('/aiml/project/DFDC/Outputs/models/model_all-18663-20-fine_adam_best.pth')
plotAccuracyAndLoss(accuracies, accuraciesTrain, f1s, loss_values)
print(f"Best epoch: {best_epoch}")
print(f"Best epoch acc: {accuracies[best_epoch]}")
print(f"Best epoch f1: {f1s[best_epoch]}")
