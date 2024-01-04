import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torchmetrics.functional import accuracy

# The model was not trained on C4 and G11 (false) and all the files inside TestData/positive folder (true)

class SequenceModel(nn.Module):
    def __init__ (self, n_features, n_classes, n_hidden = 64, n_layers = 3):
        super().__init__()

        self.n_hidden = n_hidden

        self.lstm = nn.LSTM(
            input_size = n_features,
            hidden_size = n_hidden,
            num_layers = n_layers,
            batch_first = True,
            dropout = 0.75
        )

        self.classifier = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)

        out = hidden[-1]
        return self.classifier(out)

# Defining the model, it is used to predict the output of the neural network
class CovidPredictor(pl.LightningModule):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.model = SequenceModel(n_features, n_classes)
        self.criterion = nn.CrossEntropyLoss()

    # Defining the forward pass, it is used to predict the output of the neural network
    def forward(self, x, labels = None):
        output = self.model(x)
        loss = 0
        
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    # Defining the training step, it is used to train the neural network
    def training_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss, outputs = self(sequences, labels)
        prediction = torch.argmax(outputs, dim=1)
        step_accuracy = accuracy(prediction, labels, num_classes= 2, task="multiclass")

        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_accuracy", step_accuracy, prog_bar=True, logger=True)

        return {"loss": loss, "accuracy": step_accuracy}

    # Defining the validation step, it is used to evaluate the model on the validation set
    def validation_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss, outputs = self(sequences, labels)
        prediction = torch.argmax(outputs, dim=1)
        step_accuracy = accuracy(prediction, labels, num_classes=2, task="multiclass")

        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_accuracy", step_accuracy, prog_bar=True, logger=True)

        return {"loss": loss, "accuracy": step_accuracy}

    # Defining the test step, it is used to test the model on the test dataset
    def test_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss, outputs = self(sequences, labels)
        prediction = torch.argmax(outputs, dim=1)
        step_accuracy = accuracy(prediction, labels, num_classes=n_classes, task="multiclass")

        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_accuracy", step_accuracy, prog_bar=True, logger=True)

        return {"loss": loss, "accuracy": step_accuracy}
    
    # Defining the optimizer, it is used to update the weights of the neural network
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr = 0.0001)

def Resample(pd_dataframe):
   # Assuming x_train is your DataFrame and 'name' is the column with the names
   grouped = pd_dataframe.groupby('name')
   def adjust_length(group):
      # if the group has less than 61 rows (0-30 seconds)
      if len(group) < 61:
         min_time = group['time'].min()
         max_time = group['time'].max()
         # Interpolate to get 60 rows, the time should be 0, 0.5, 1, 1.5, 2, 2.5, ..., 29.5, 30 and the change_in_resistance should be interpolated linearly
         f = interp1d(group['time'], group['change_in_resistance'], fill_value="extrapolate")
         time = np.arange(0, 30.5, 0.5)
         change_in_resistance = f(time)
         # Normalize the change_in_resistance values to be between 0 and 1
         change_in_resistance = (change_in_resistance - change_in_resistance.min()) / (change_in_resistance.max() - change_in_resistance.min())
         # Return a DataFrame with the interpolated values
         return pd.DataFrame({'name' : group['name'].unique()[0], 'time': time, 'change_in_resistance': change_in_resistance})
      elif len(group) > 61:
         min_time = group['time'].min()
         max_time = group['time'].max()
         # Downsample the group to 60 rows, the time should be 0, 0.5, 1, 1.5, 2, 2.5, ..., 29.5, 30 and the change_in_resistance should be interpolated linearly
         f = interp1d(group['time'], group['change_in_resistance'])
         time = np.arange(0, 30.5, 0.5)
         change_in_resistance = f(time)
         # Normalize the change_in_resistance values to be between 0 and 1
         change_in_resistance = (change_in_resistance - change_in_resistance.min()) / (change_in_resistance.max() - change_in_resistance.min())
         # Return a DataFrame with the interpolated values
         return pd.DataFrame({'name' : group['name'].unique()[0], 'time': time, 'change_in_resistance': change_in_resistance})
      elif len(group) == 61:
         # Normalize the change_in_resistance values to be between 0 and 1
         change_in_resistance = (group['change_in_resistance'] - group['change_in_resistance'].min()) / (group['change_in_resistance'].max() - group['change_in_resistance'].min())
         # Return the original DataFrame
         return pd.DataFrame({'name' : group['name'].unique()[0], 'time': group['time'], 'change_in_resistance': change_in_resistance})

   # Apply the adjust_length function to each group of the DataFrame
   pd_dataframe = grouped.apply(adjust_length).reset_index(drop=True)
   # Return the new DataFrame
   return pd_dataframe

def plot_sequence(sequence, prediction, name):
    plt.figure(figsize=(10, 4))
    plt.plot(sequence['time'], sequence['change_in_resistance'], label='Change in Resistance')
    plt.title(f"Test Sequence for {name} - Predicted Label: {prediction}")
    plt.xlabel("Time (s)")
    plt.ylabel("Change in Resistance")
    plt.legend()
    st.pyplot(plt)

ModelPath = 'Models/lstm_model.pt'
model = CovidPredictor(1, 2)

PREDICTION = []
FEATURE_COLUMN = ['change_in_resistance']

def run_model_on_df(df_test):
    checkpoint_path = "Models/checkpoint.ckpt"
    model = CovidPredictor.load_from_checkpoint(
        checkpoint_path,
        n_features = 1,
        n_classes = 2
    )

    #creating a sequence from the df_test dataframe
    sequences = []
    for name, group in df_test.groupby("name"):
        sequence_feature = group[FEATURE_COLUMN]
        sequences.append(sequence_feature)

    class CovidDataset(Dataset):
        def __init__(self, sequences):
            self.sequences = sequences
            self.max_length = max(len(sequence) for sequence in sequences)

        def __len__(self):
            return len(self.sequences)
        
        def __getitem__(self, idx):
            sequence = self.sequences[idx]
            return dict(
                sequence = torch.Tensor(sequence.to_numpy())
            )

    test_dataset = CovidDataset(sequences)
    names = df_test['name'].unique()
    n = 0

    #predicting the labels for the test_dataset
    prediction = []

    for item in tqdm (test_dataset):
        sequence = item["sequence"]

        # Move the sequence tensor to the same device as the model
        sequence = sequence.to(model.device)

        _, output = model(sequence.unsqueeze(dim = 0))
        predictions = torch.argmax(output, dim = 1)
        prediction.append(predictions.item())
        st.write(str(str(names[n]) + " - Predicted Label is: " + str(predictions.item())))
        PREDICTION.append(predictions.item())
        print(str(names[n]) + " - Prediction is: " + str(predictions.item())) # debug

        # Convert the sequence tensor back to numpy for plotting
        sequence_array = sequence.cpu().numpy().squeeze()
        sequence_df = pd.DataFrame(sequence_array, columns = [FEATURE_COLUMN])
        sequence_df["time"] = np.arange(0, 30.5, 0.5)
        plot_sequence(sequence_df, PREDICTION[n], names[n])
        n += 1
        st.divider()

st.title('Fentanyl Detection - Signal Analysis 📊')

# Asking user to upload a excel or csv file
uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
if uploaded_file is not None:
   # If the uploaded file is a csv file
    if uploaded_file.name.split('.')[1] == 'csv':
        # Read the csv file into a Pandas DataFrame
        x_test = pd.read_csv(uploaded_file)
        # Resample the DataFrame
        x_test = Resample(x_test)
    elif uploaded_file.name.split('.')[1] == 'xlsx':
        # Read the excel file into a Pandas DataFrame
        x_test = pd.read_excel(uploaded_file)
        # Resample the DataFrame
        x_test = Resample(x_test)

# Create a button to run the model
if st.button('Run Model'):
   # If the user has uploaded a file
    if uploaded_file is not None:
        # Run the model on the DataFrame
        run_model_on_df(x_test)
    else:
        # Print an error message
        st.write('Please upload a file')
