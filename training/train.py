import torch 
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F


def compute_accuracies(predictions, y):
    """Compute the accuracy of the prediction based on lable y"""
    return np.mean(np.equal(predictions.numpy(), y.numpy()))


def train_model(train_data, test_data, model, lr=0.1, momentum=0.9, nesterov=False, n_epochs=30, path="LeNet1989.pt"):
    """
    Trains the model for N epochs given train_data and hyper-parameters    
    Params:
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov)

    # batchify the data
    train_data = DataLoader(dataset=train_data, batch_size= 32, shuffle=True)
    test_data = DataLoader(dataset=test_data, batch_size= 32, shuffle=True)

    # ideally defining a dataframe here that can be used for tracking and plotting after. 
    # save: epoch | train_loss | train_accuracy | validation_loss | validation_accuracy

    for epoch in range(1, n_epochs + 1):
        print("---------\nEpoch: {}\n".format(epoch))

        # run the training cycle 
        train_loss, train_acc = run_epoch(data=train_data, model=model.train(), optimizer=optimizer)
        print('Train | Loss: {:.6f}  Accuracy: {:.6f}'.format(train_loss, train_acc))

        # run the validation cycle
        val_loss, val_acc = run_epoch(data=test_data, model=model.eval(), optimizer=optimizer)
        #TODO: Add print statements with validation loss
        print('Validation | Loss: {:.6f}  Accuracy: {:.6f}'.format(val_loss, val_acc))

        # save model
        torch.save(model, path)






def run_epoch(data, model, optimizer):
    """
    Runs a singular epoch on the passed model.
    Params:
        data:
        model:
        optimizer

    Returns:
        loss: 
        accuracy:
    """ 

    losses = []
    batch_accuracies = []

    # determine if train-mode
    is_training = model.training

    for x, y in tqdm(data):

        # feed through network and compute accuracy
        out = model(x)
        prediction = torch.argmax(out, dim=1)
        batch_accuracies.append(compute_accuracies(prediction, y))

        y_one_hot = F.one_hot(y, num_classes=10).float()

        # compute loss according to the paper
        loss = F.mse_loss(out, y_one_hot)

        losses.append(loss.data.item())


        # If training, update weights
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



    # after batch loop compute average loss and average accuracy and return 
    avg_loss = np.mean(losses)
    avg_acc = np.mean(batch_accuracies)

    return avg_loss, avg_acc





