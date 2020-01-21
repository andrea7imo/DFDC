import copy
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.rmsprop import RMSprop
from torch.optim.adamax import Adamax
from torch.backends import cudnn
from tqdm import tqdm

from network.xception import Xception

DEVICE = 'cuda'

BATCH_SIZE = 5

LR = 1e-2
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-5

NUM_EPOCHS = 30
STEP_SIZE = 10
GAMMA = 0.1

LOG_FREQUENCY = 10

OPTM_HYPER = True
alpha = 1

# Per salvare i valori di hyperparameters durante l'ottimizzazione
def saveHyperparameter(accuracy, PATH):
    torch.save({
        'accuracy': accuracy,
        'LR': LR,
        'WEIGHT_DECAY': WEIGHT_DECAY,
        'STEP_SIZE': STEP_SIZE
    }, PATH)


def loadHypeparameter(PATH):
    checkpoint = torch.load(PATH)
    accuracy = checkpoint['accuracy']
    LR = checkpoint['LR']
    WEIGHT_DECAY = checkpoint['WEIGHT_DECAY']
    STEP_SIZE = checkpoint['STEP_SIZE']
    return accuracy, LR, WEIGHT_DECAY, STEP_SIZE

# Per salvare il modello una volta finito il training

def saveModel(best_epoch, best_model_wts, loss_values, accuracies, accuraciesTrain, PATH):
    torch.save({
              'best_epoch': best_epoch,
              'model_state_dict': best_model_wts,
              'loss_values': loss_values,
              'accuracies': accuracies,
              'accuraciesTrain': accuraciesTrain
              }, PATH)

def loadModel(PATH):
    net = Xception()
    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    best_model_wts = checkpoint['model_state_dict']
    best_epoch = checkpoint['best_epoch']
    loss_values = checkpoint['loss_values']
    accuracies = checkpoint['accuracies']
    accuraciesTrain = checkpoint['accuraciesTrain']
    return net, best_epoch, loss_values, accuracies, accuraciesTrain, best_model_wts

def loadModelDeepForensics():
    model = torch.load('/aiml/references/faceforensics++_models_subset/full/xception/full_c23.p')
    model = model.cuda()
    return model

# Plot dell' accuracy e della loss function

def plotAccuracyAndLoss(accuracies, accuraciesTrain, loss_values):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    plt.title('The accuracy', fontsize=20)
    ax.plot(accuracies, 'r', label='Validation set')
    ax.plot(accuraciesTrain, 'b', label='Training set')
    ax.set_xlabel(r'Epoch', fontsize=10)
    ax.set_ylabel(r'Accuracy', fontsize=10)
    ax.legend()
    ax.tick_params(labelsize=20)
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    plt.title('The loss function', fontsize=20)
    ax.plot(loss_values, 'b', label='Loss values')
    ax.set_xlabel(r'Epoch', fontsize=10)
    ax.set_ylabel(r'Loss', fontsize=10)
    ax.legend()
    ax.tick_params(labelsize=20)
    plt.show()

# Preparazione per il train

def prepareTraining(net, type_optimizer):
    criterion = nn.CrossEntropyLoss()
    parameters_to_optimize = net.parameters()
    if type_optimizer == 'SGD':
        optimizer = optim.SGD(parameters_to_optimize, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    if type_optimizer == 'Adam':
        optimizer = optim.Adam(parameters_to_optimize,lr=LR,weight_decay=WEIGHT_DECAY)
    if type_optimizer == 'RMSprop':
        optimizer = RMSprop(parameters_to_optimize,lr=LR,weight_decay=WEIGHT_DECAY)
    if type_optimizer == 'Adamax':
        #Implements Adamax algorithm (a variant of Adam based on infinity norm).
        optimizer = Adamax(parameters_to_optimize,lr=LR,weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    return  criterion, optimizer, scheduler

# confusion matrix
def confusion(prediction, truth):
    confusion_vector = prediction / truth

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector >= 4294967295).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives

# Train

def train(net, tr_dataloader, val_dataloader,type_optimizer):
    net = net.to(DEVICE)

    cudnn.benchmark
    criterion, optimizer, scheduler = prepareTraining(net,type_optimizer)
    current_step = 0
    bestAccuracy = 0.0
    accuracies = []
    accuraciesTrain = []
    loss_values = []

    for epoch in range(NUM_EPOCHS):
        print('Starting epoch {}/{}, LR = {}'.format(epoch + 1, NUM_EPOCHS, scheduler.get_lr()))

        for images, labels in tr_dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            net.train()
            optimizer.zero_grad()

            outputs = net(images)

            loss = criterion(outputs, labels)

            if current_step % LOG_FREQUENCY == 0:
                print('Step {}, Loss {}'.format(current_step, loss.item()))

            loss.backward()
            optimizer.step()

            current_step += 1

        scheduler.step()

        # validation
        print("\n\tValidation:")
        accuracy = test(net, val_dataloader)
        if OPTM_HYPER == False:
            # test on the training set
            accuracyTrain = test(net, tr_dataloader)
            print(f"\tThe accuracy on validation set: {accuracy}")
            accuracies.append(accuracy)
            accuraciesTrain.append(accuracyTrain)
            loss_values.append(loss.item())

        if accuracy > bestAccuracy:
            bestAccuracy = accuracy
            if OPTM_HYPER == False:
                best_model_wts = copy.deepcopy(net.state_dict())
                best_epoch = epoch

    if OPTM_HYPER == False:
        plotAccuracyAndLoss(accuracies, accuraciesTrain, loss_values)

        print(f"The best value of accuracy is: {bestAccuracy} \nThe best epoch is: {best_epoch}")
        return best_epoch, best_model_wts, bestAccuracy, accuracies, accuraciesTrain, loss_values
    else:
        return bestAccuracy

# Test

def test(net, test_dataloader):
    net = net.to(DEVICE)
    net.train(False)

    running_corrects = 0
    i = 0
    for images, labels in tqdm(test_dataloader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = net(images)

        _, preds = torch.max(outputs.data, 1)

        if i == 0:
            i += 1
            prediction = preds
            truth = labels.data
        else:
            prediction = torch.cat((prediction, preds), dim=0)
            truth = torch.cat((truth, labels.data), dim=0)
        running_corrects += torch.sum(preds == labels.data).data.item()

    accuracy = running_corrects / float(len(test_dataloader.dataset))
    true_positives, false_positives, true_negatives, false_negatives = confusion(prediction, truth)
    precision = true_positives/(true_positives + alpha*false_positives)
    recall = true_positives/(true_positives + false_negatives)
    F_1 = 2*precision*recall/(precision+recall)
    print('Test Accuracy: {}'.format(accuracy))
    print(f'Precision: {F_1}')
    return accuracy

# Random search

def randomSearchCoarse(train_dataloader, validation_dataloader,type_optimizer):
    path_init = '/aiml/project/utility/coarse/opt_hyper_coarse_'         #path dove salvere i risultati
    bestAccuracy = 0
    global NUM_EPOCHS, OPTM_HYPER
    NUM_EPOCHS = 5
    OPTM_HYPER = True

    for i in range(1):
        global LR, WEIGHT_DECAY
        global criterion, optimizer, scheduler

        LR = 10**random.uniform(-3, -6)
        WEIGHT_DECAY = 10**random.uniform(-5, -1)
        #STEP_SIZE = round(random.uniform(5, 30)) DA ottimizzare nella fase di fine!!!
        print(f"[{i}/16]: \tLR: {LR} \tWEIGHT_DECAY: {WEIGHT_DECAY}")
        print(f"****************************** START TRAINING ******************************")

        model = loadModelDeepForensics()
        criterion, optimizer, scheduler = prepareTraining(model,type_optimizer)

        bestAccuracy = train(model, train_dataloader, validation_dataloader)

        path = path_init + str(i)
        print(f"\t\tAccuracy: {bestAccuracy}")
        saveHyperparameter(bestAccuracy, path)
        print(f"****************************** END TRAINING ******************************")


def randomSearchFine(train_dataloader, validation_dataloader,type_optimizer):
    path_init = '/aiml/project/utility/fine/opt_hyper_fine_'  # path dove salvere i risultati
    bestAccuracy = 0
    global NUM_EPOCHS, OPTM_HYPER
    NUM_EPOCHS = 15 # o di più
    OPTM_HYPER = True

    for i in range(1):
        global LR, WEIGHT_DECAY, STEP_SIZE
        global criterionLabel, criterionDomain, optimizer, scheduler

        # Bisogna far esegure il coarse e poi questo prendendo un sottoinsieme migliore!
        LR = 10 ** random.uniform(-3, -6)
        WEIGHT_DECAY = 10 ** random.uniform(-5, -1)
        STEP_SIZE = round(random.uniform(5, 30))
        print(f"[{i}/16]: \tLR: {LR} \tWEIGHT_DECAY: {WEIGHT_DECAY} \tSTEP_SIZE: {STEP_SIZE}")
        print(f"****************************** START TRAINING ******************************")

        model = loadModelDeepForensics()
        criterionLabel, optimizer, scheduler = prepareTraining(model,type_optimizer)

        bestAccuracy = train(model, train_dataloader, validation_dataloader,type_optimizer)

        path = path_init + str(i)
        print(f"\t\tAccuracy: {bestAccuracy}")
        saveHyperparameter(bestAccuracy, path)
        print(f"****************************** END TRAINING ******************************")