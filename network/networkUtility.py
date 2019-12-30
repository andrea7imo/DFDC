import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from tqdm import tqdm

from network.xception import Xception

DEVICE = 'cuda'

BATCH_SIZE = 5       # Higher batch sizes allows for larger learning rates. An empirical heuristic suggests that, when changing
                     # the batch size, learning rate should change by the same factor to have comparable results

LR = 1e-2            # The initial Learning Rate
MOMENTUM = 0.9       # Hyperparameter for SGD, keep this at 0.9 when using SGD
WEIGHT_DECAY = 5e-5  # Regularization, you can keep this at the default

NUM_EPOCHS = 30      # Total number of training epochs (iterations over dataset)
STEP_SIZE = 10       # How many epochs before decreasing learning rate (if using a step-down policy)
GAMMA = 0.1          # Multiplicative factor for learning rate step-down

LOG_FREQUENCY = 10
# Per salvare i valori di hyperparameters durante l'ottimizzazione
def saveHyperparameter(accuracy, LR, WEIGHT_DECAY, STEP_SIZE, PATH):
    torch.save({
        'accuracy': accuracy,
        'LR': LR,
        'WEIGHT_DECAY': WEIGHT_DECAY,
        'STEP_SIZE': STEP_SIZE,
    }, PATH)


def loadHypeparameter(PATH):
    checkpoint = torch.load(PATH)
    accuracy = checkpoint['accuracy']
    LR = checkpoint['LR']
    WEIGHT_DECAY = checkpoint['WEIGHT_DECAY']
    STEP_SIZE = checkpoint['STEP_SIZE']
    alpha = checkpoint['alpha']
    return accuracy, LR, WEIGHT_DECAY, STEP_SIZE, alpha

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

# Plot dell' accuracy e della loss function

def plotAccuracyAndLoss(accurancies, accuranciesTrain, loss_values):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    plt.title('The accurancy', fontsize=20)
    ax.plot(accurancies, 'r', label='Validation set')
    ax.plot(accuranciesTrain, 'b', label='Training set')
    ax.set_xlabel(r'Epoch', fontsize=10)
    ax.set_ylabel(r'Accurancy', fontsize=10)
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

def prepareTraining(net):
    criterion = nn.CrossEntropyLoss()
    parameters_to_optimize = net.parameters()

    optimizer = optim.SGD(parameters_to_optimize, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    return  criterion, optimizer, scheduler

# Train

def train(net, tr_dataloader, val_dataloader):
    net = net.to(DEVICE)

    cudnn.benchmark
    criterion, optimizer, scheduler = prepareTraining(net)
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
        # test on the training set
        accuracyTrain = test(net, tr_dataloader)
        print(f"\tThe accuracy on validation set: {accuracy}")
        accuracies.append(accuracy)
        accuraciesTrain.append(accuracyTrain)
        loss_values.append(loss.item())

        if accuracy > bestAccuracy:
            bestAccuracy = accuracy
            best_model_wts = copy.deepcopy(net.state_dict())
            best_epoch = epoch

    plotAccuracyAndLoss(accuracies, accuraciesTrain, loss_values)

    print(f"The best value of accuracy is: {bestAccuracy} \nThe best epoch is: {best_epoch}")
    return best_epoch, best_model_wts, bestAccuracy, accuracies, accuraciesTrain, loss_values

# Test

def test(net, test_dataloader):
    net = net.to(DEVICE)
    net.train(False)

    running_corrects = 0
    for images, labels in tqdm(test_dataloader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = net(images)

        _, preds = torch.max(outputs.data, 1)

        running_corrects += torch.sum(preds == labels.data).data.item()

    accuracy = running_corrects / float(len(test_dataloader.dataset))

    print('Test Accuracy: {}'.format(accuracy))
    return accuracy