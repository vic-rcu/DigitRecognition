import torch 
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
import time

from training.train import compute_accuracies, run_epoch_ModCNN


def tune_ModCNN(train_data, test_data, model, n_epochs=15, path="ModCNN.pt"):
        """
        Trains the model for N epochs given train_data and hyper-parameters    
        Params:

        Returns:
            df:
            all_preds:
            all_labels
        """
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=10,      
            eta_min=1e-5   
        )

        columns = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_time", "gen_gap"]

        df = pd.DataFrame(columns=columns)

        # batchify the data
        train_data = DataLoader(dataset=train_data, batch_size= 64, shuffle=True)
        test_data = DataLoader(dataset=test_data, batch_size= 64, shuffle=True)

        # ideally defining a dataframe here that can be used for tracking and plotting after. 
        # save: epoch | train_loss | train_accuracy | validation_loss | validation_accuracy

        for epoch in range(1, n_epochs + 1):
            start_time = time.time()

            print("---------\nEpoch: {}\n".format(epoch))

            # run the training cycle 
            train_loss, train_acc = run_epoch_ModCNN(data=train_data, model=model.train(), optimizer=optimizer)
            print('Train | Loss: {:.6f}  Accuracy: {:.6f}'.format(train_loss, train_acc))

            # run the validation cycle
            val_loss, val_acc = run_epoch_ModCNN(data=test_data, model=model.eval(), optimizer=optimizer)
            #TODO: Add print statements with validation loss
            print('Validation | Loss: {:.6f}  Accuracy: {:.6f}'.format(val_loss, val_acc))

            epoch_time = time.time() - start_time

            df.loc[len(df)] = [epoch, train_loss, train_acc, val_loss, val_acc, epoch_time, train_acc - val_acc]

            # save model
            torch.save(model, path)

            scheduler.step()
        

        return df