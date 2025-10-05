# Setting path for proper imports 
import sys
import os
sys.path.append(os.path.abspath('..'))

# Importing all Relevant Modules
import data.utils as data
import training.train as train
from models.LeNet1989 import LeNet1989


model = LeNet1989()

(x_train, y_train), (x_test, y_test) = data.get_prepared_data()

train_set = data.create_dataset(x_train, y_train)
test_set = data.create_dataset(x_test, y_test)

df = train.train_LeNet1989(train_data = train_set, test_data=test_set, model=le_net, n_epochs=30)

print(df)
