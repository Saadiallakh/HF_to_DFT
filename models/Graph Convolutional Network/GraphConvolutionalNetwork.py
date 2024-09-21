import os
import joblib
import optuna

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold 
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score


#----------# Data Loading #----------#


#-# Check if the GPU is available on the device
#   If not available, the process will run on CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-# Load data from a CSV file
df = pd.read_csv("./NN_ML.csv", delimiter=",")

def sanitize_molecule(mol):
    """
    Function to call the Chem.SanitizeMol(mol) method to check and fix structural errors in a molecule.
    
    Parameters:
    - mol: RDKit molecule object.
    
    Returns:
    - mol: RDKit molecule object if the check is successful.
    - None: If an exception occurs during the check.
    """
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception as e:
        print(f"Failed to sanitize molecule: {str(e)}")
        return None

def read_sdf_files(directory):
    """
    Function to read SDF (Structure Data File) files in a directory.
    
    Parameters:
    - directory: Path to the directory with SDF files.
    
    Returns:
    - molecules: List of tuples (file_name, sanitized_mol) for successfully read molecules.
    """
    molecules = []
    failed_files = []
    
    for file_name in os.listdir(directory):
        if file_name.endswith('.sdf'):
            file_path = os.path.join(directory, file_name)
            try:
                suppl = Chem.SDMolSupplier(file_path)
                for mol in suppl:
                    if mol is not None:
                        sanitized_mol = sanitize_molecule(mol)
                        if sanitized_mol is not None:
                            molecules.append((file_name, sanitized_mol))
            except Exception as e:
                print(f"Failed to read {file_name}: {str(e)}")
                failed_files.append(file_name)
    
    if failed_files:
        print(f"Failed to read {len(failed_files)} files.")
    
    return molecules

def compute_rdkit_features(mol):
    """
    Function to compute molecule features (fingerprints) using the RDKit library.
    
    Parameters:
    - mol: RDKit molecule object.
    
    Returns:
    - np.array: Array of molecule features.
    """
    if mol is None:
        return None
    morgan_fp = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=256)
    return np.array(morgan_fp)

#-# Read structures from SDF files
sdf_directory = "./sdf"
molecules = read_sdf_files(sdf_directory)

#-# Extract properties and map them to DataFrame
mol_features = {}
mol_objects = {}
for file_name, mol in molecules:
    mol_name = os.path.splitext(file_name)[0] 
    features = compute_rdkit_features(mol)
    mol_features[mol_name] = features
    mol_objects[mol_name] = mol

df['rdkit_features'] = df['name'].map(mol_features)
df['rdkit_molecules'] = df['name'].map(mol_objects)

print(df.head())

def create_graph_data(df):
    """
    Function to create a graph representation of the dataset.
    
    Parameters:
    - df: Pandas DataFrame containing data about molecules and their chemical properties.
    
    Returns:
    - graph_data_list: List of Data objects representing the graph data.
    """
    graph_data_list = []
    
    for idx, row in df.iterrows():
        rdkit_features = row['rdkit_features']
        rdkit_mol = row['rdkit_molecules']
        if rdkit_features is None or rdkit_mol is None:
            continue  

        rdkit_mol = Chem.AddHs(rdkit_mol)
        num_atoms = rdkit_mol.GetNumAtoms()
        
        hf_features = row[['hf_gibbs_free_energy_ev', 'hf_electronic_energy_ev', 'hf_entropy_ev',
                           'hf_enthalpy_ev', 'hf_dipole_moment_d', 'hf_gap_ev']].astype(np.float64).values
        
        node_features_list = []
        
        for atom_idx in range(num_atoms):
            atom = rdkit_mol.GetAtomWithIdx(atom_idx)
            
            atom_features = [
                atom.GetAtomicNum(),            #-# Atomic number
                atom.GetExplicitValence(),      #-# Valence       
                atom.GetFormalCharge(),         #-# Formal charge       
                atom.GetIsAromatic(),           #-# Aromaticity       
                atom.GetMass()                  #-# Mass
            ]
            
            combined_features = np.concatenate([hf_features, rdkit_features, atom_features])
            node_features_list.append(combined_features)
        
        node_features = np.array(node_features_list, dtype=np.float64)
        node_features = torch.tensor(node_features, dtype=torch.float64).to(device)
        
        edge_index = []
        for bond in rdkit_mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
        
        graph_data = Data(x=node_features, edge_index=edge_index)
        graph_data.y = torch.tensor(row[['dft_gibbs_free_energy_ev', 'dft_electronic_energy_ev', 'dft_entropy_ev',
                                         'dft_enthalpy_ev', 'dft_dipole_moment_d', 'dft_gap_ev']].astype(np.float64).values, dtype=torch.float64).to(device)
        graph_data_list.append(graph_data)
    
    return graph_data_list

graph_data_list = create_graph_data(df)


#----------# Normalization and Dataset Splitting #----------#


#-# Create a list of original indices for further mapping
original_indices = np.arange(len(graph_data_list))

#-# train_test_split is used to split the graph_data_list containing Data objects for graphs
#   into training (train_val_graphs) and test (test_graphs) sets, and also to track the original indices of observations
train_val_graphs, test_graphs, train_val_indices, test_indices = train_test_split(
    graph_data_list, original_indices, test_size=0.1, random_state=42)

#-# train_val_graphs (previously obtained train_val_graphs) is split into training (train_graphs) and validation (val_graphs) sets 
#   Their indices are also tracked
train_graphs, val_graphs, train_indices, val_indices = train_test_split(
    train_val_graphs, train_val_indices, test_size=0.1, random_state=42)

#-# Extract node features (graph.x.numpy()) from the training set train_graphs, representing the node feature matrices for each graph
#   np.vstack is used to vertically concatenate these feature matrices into one large node_features matrix 
node_features = np.vstack([graph.x.cpu().numpy() for graph in train_graphs])

#-# Extract target values (graph.y.numpy()) from the training set train_graphs, representing the target value matrices for each graph
#   np.vstack is used to vertically concatenate these target value matrices into one large target_values matrix 
target_values = np.vstack([graph.y.cpu().numpy() for graph in train_graphs])

#-# Use StandardScaler() to normalize the node features
feature_scaler = StandardScaler().fit(node_features)

#-# Use MinMaxScaler() to normalize the target values to the 0 to 1 range 
target_scaler = MinMaxScaler().fit(target_values)

#-# Save the scalers
scaler_dir = "./output"

feature_scaler_path = os.path.join(scaler_dir, "feature_scaler.pkl")
target_scaler_path = os.path.join(scaler_dir, "target_scaler.pkl")

joblib.dump(feature_scaler, feature_scaler_path)
joblib.dump(target_scaler, target_scaler_path)

print(f"Scalers saved to {scaler_dir}")

#-# Transformation
def transform_graphs(graphs, feature_scaler, target_scaler):
    """
    Function to transform the features (graph.x) and target variables (graph.y) in the graphs used in the neural network.

    Parameters:
    - graphs: List of Data objects.
    - feature_scaler: Scaler for the features.
    - target_scaler: Scaler for the target values.

    Returns:
    - None: Transforms features and targets in place.
    """
    for graph in graphs:
        graph.x = torch.tensor(feature_scaler.transform(graph.x.cpu().numpy()), dtype=torch.float64).to(device)
        graph.y = torch.tensor(target_scaler.transform(graph.y.cpu().numpy().reshape(1, -1)), dtype=torch.float64).view(-1).to(device)

transform_graphs(train_graphs, feature_scaler, target_scaler)
transform_graphs(val_graphs, feature_scaler, target_scaler)
transform_graphs(test_graphs, feature_scaler, target_scaler)


#----------# Graph Convolutional Network #----------#


class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) using PyTorch.

    This class defines a GCN with multiple GCNConv layers followed by a fully connected layer.

    Parameters:
    - input_dim (int): Number of input features per node.
    - hidden_dim (int): Number of features in hidden layers.
    - output_dim (int): Number of output features.
    - num_layers (int): Number of GCNConv layers.
    - activation_function (str): Activation function name (e.g., 'relu').

    Methods:
    - forward(data): Executes a forward pass through the network.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation_function, dropout_rate):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()                                #-# List to hold GCNConv layers
        self.convs.append(GCNConv(input_dim, hidden_dim))           #-# Initial GCNConv layer
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))      #-# Additional GCNConv layers
        self.fc = nn.Linear(hidden_dim, output_dim)                 #-# Fully connected layer for output
        self.activation_function = activation_function              #-# Activation function for layers
        self.dropout = nn.Dropout(p=dropout_rate)                   #-# Add Dropout layer for regularization
        self.float()                                                

    def forward(self, data):
        """
        Forward pass through the GCN model.

        Parameters:
        - data (Data): A PyTorch Geometric Data object containing:
          - x (Tensor): Node features.
          - edge_index (Tensor): Graph connectivity in COO format.

        Returns:
        - Tensor: Output feature tensor after passing through all layers.
        """
        x, edge_index = data.x, data.edge_index
        
        #-# Apply GCNConv layers and activation function
        for conv in self.convs:
            x = conv(x, edge_index)
            x = getattr(F, self.activation_function)(x)
            x = self.dropout(x)
        
        #-# Aggregate node features
        x = torch.mean(x, dim=0)

        #-# Pass through the fully connected layer
        x = self.fc(x)

        return x

#-# Set the loss function
criterion = nn.MSELoss()

#-# Set the dimensions of the input and output layers 
input_dim = node_features.shape[1]
output_dim = 6
        
num_epochs = 1000         #-# Sets the total number of times the model will process the entire dataset to update weights and reduce error
patience = 20             #-# Sets the number of epochs to tolerate without improvement in validation metric (early stopping)
min_delta = 0.0001        #-# Minimum change required in validation loss to be considered as an improvement

#-# Define the objective function for Optuna optimization
def objective(trial):
    """
    Objective function for Optuna that defines the task for hyperparameter search.

    Parameters:
    - trial (optuna.trial.Trial): Optuna trial object used to suggest hyperparameters.

    Returns:
    - float: Mean MSE value across all folds of cross-validation.
    """
    #-# Set the hyperparameter grid
    hidden_dim = trial.suggest_int('hidden_dim', 5, 200, log=True)
    num_layers = trial.suggest_int('num_layers', 2, 20)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop', 'NAdam', 'RAdam', 'AdamW'])
    activation_function = trial.suggest_categorical('activation_function', ['relu', 'sigmoid', 'tanh', 'leaky_relu'])

    #-# Create a KFold object for 3-fold cross-validation with data shuffling
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    val_losses = []

    #-# Loop through the data splits
    for train_index, val_index in kf.split(train_graphs):
        train_graphs_cv = [train_graphs[i] for i in train_index]
        val_graphs_cv = [train_graphs[i] for i in val_index]
        
        #-# Create a model with the current hyperparameter values
        model = GCN(input_dim, hidden_dim, output_dim, num_layers, activation_function, dropout_rate).double()
        model.to(device)

        #-# Choose optimizer
        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'NAdam':
            optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'RAdam':
            optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        #-# Counters for EarlyStopping
        #   best_val_loss:     Initial value for the variable that will store the best (lowest) validation loss 
        #   epochs_no_improve: Counter for epochs without improvement. Once this counter reaches the patience value, training is stopped
        best_val_loss = float('inf')
        epochs_no_improve = 0

        #-# Train the model
        for epoch in range(num_epochs):
            model.train()
            for data in train_graphs_cv:
                data = data.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, data.y)
                loss.backward()
                optimizer.step()

            #-# Evaluate on the validation set
            model.eval()
            val_epoch_losses = []
            for data in val_graphs_cv:
                data = data.to(device)
                with torch.no_grad():
                    output = model(data)
                    val_loss = mean_squared_error(data.y.cpu().numpy(), output.cpu().numpy())
                    val_epoch_losses.append(val_loss)

            avg_val_loss = np.mean(val_epoch_losses)

            #-# Apply EarlyTopping technique to prevent overfitting
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                break
        
        val_losses.append(best_val_loss)

    return np.mean(val_losses)

#-# Run the optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1000)

#-# Print the best hyperparameters
best_trial = study.best_trial
print(f'Best trial value: {best_trial.value}')
print(f'Best hyperparameters: {best_trial.params}')

#-# Save the optimization results
with open('./optimization_results.txt', 'w') as f:
    f.write(f'Best trial value: {best_trial.value}\n')
    f.write(f'Best hyperparameters: {best_trial.params}\n')

    f.write('\nAll trial results:\n')
    for trial in study.trials:
        f.write(f'Trial {trial.number}: Value={trial.value}, Params={trial.params}\n')

#-# Print the best hyperparameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)

#-# Get the best hyperparameters
best_hidden_dim = best_params['hidden_dim']
best_num_layers = best_params['num_layers']
best_learning_rate = best_params['learning_rate']
best_weight_decay = best_params['weight_decay']
best_dropout_rate = best_params['dropout_rate']
best_optimizer_name = best_params['optimizer']
best_activation_function = best_params['activation_function']

#-# Create the model with best hyperparameters 
#   Model is initialized with double() for double precision
best_model = GCN(input_dim, best_hidden_dim, output_dim, best_num_layers, best_activation_function, best_dropout_rate).double()
best_model.to(device)

#-# Get the best optimizer
best_optimizer = None
if best_optimizer_name == 'Adam':
    best_optimizer = torch.optim.Adam(best_model.parameters(), lr=best_learning_rate, weight_decay=best_weight_decay)
elif best_optimizer_name == 'SGD':
    best_optimizer = torch.optim.SGD(best_model.parameters(), lr=best_learning_rate, weight_decay=best_weight_decay)
elif best_optimizer_name == 'RMSprop':
    best_optimizer = torch.optim.RMSprop(best_model.parameters(), lr=best_learning_rate, weight_decay=best_weight_decay)
elif best_optimizer_name == 'NAdam':
    best_optimizer = torch.optim.NAdam(best_model.parameters(), lr=best_learning_rate, weight_decay=best_weight_decay)
elif best_optimizer_name == 'RAdam':
    best_optimizer = torch.optim.RAdam(best_model.parameters(), lr=best_learning_rate, weight_decay=best_weight_decay)
elif best_optimizer_name == 'AdamW':
    best_optimizer = torch.optim.AdamW(best_model.parameters(), lr=best_learning_rate, weight_decay=best_weight_decay)
else:
    raise ValueError(f"Unsupported optimizer: {best_optimizer_name}")

#-# Variables to track the best model
#   best_model_weights is initialized as None to store the best model weights
#   best_val_loss is initialized as infinity (float('inf')) to track the best validation loss
#   epochs_no_improve is initialized as 0 to track the number of epochs without improvements
best_model_weights = None
best_val_loss = float('inf')
epochs_no_improve = 0

#-# Create empty lists to append results 
train_losses = []
val_losses = []          

#-# total_loss computes the total loss on all training graphs
#-# In each epoch, the mean squared error on the validation set (val_graphs) is also computed
#-# train_losses and val_losses store the average loss for each epoch for later plotting of training curves
for epoch in range(num_epochs):
    total_loss = 0
    for data in train_graphs:
        data = data.to(device)
        best_model.train()
        best_optimizer.zero_grad()
        output = best_model(data)
        loss = criterion(output, data.y)
        loss.backward()
        best_optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_graphs)
    train_losses.append(avg_train_loss)
    
    val_epoch_losses = []
    for data in val_graphs:
        data = data.to(device)
        best_model.eval()
        with torch.no_grad():
            output = best_model(data)
            val_loss = mean_squared_error(data.y.cpu().numpy(), output.cpu().numpy())
            val_epoch_losses.append(val_loss)
    
    #-# If validation loss (avg_val_loss) improves (less than best_val_loss - min_delta), update best_val_loss 
    #   epochs_no_improve is reset to 0, and best model weights (best_model_weights) are saved
    avg_val_loss = np.mean(val_epoch_losses)
    val_losses.append(avg_val_loss)
    
    print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    if avg_val_loss < best_val_loss - min_delta:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        best_model_weights = best_model.state_dict()
    else:
        epochs_no_improve += 1
    
    if epochs_no_improve == patience:
        print(f'Early stopping after {epoch + 1} epochs.')
        break

#-# If best model weights were saved (best_model_weights is not None), load them back into the best_model
if best_model_weights is not None:
    best_model.load_state_dict(best_model_weights)

#-# Plot Train & Val curves
#   Save them as training_validation_loss.png
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()

plot_path = "./training_validation_loss.png"  
plt.savefig(plot_path)

print(f"Saved Train vs. Val plot to {plot_path}")

#-# Save the trained model
model_path = "./best_gcn_model.pt"

#-# model_info creates a dictionary containing information about the model parameters to be saved 
model_info = {
    'input_dim': input_dim,
    'hidden_dim': best_hidden_dim,
    'output_dim': output_dim,
    'num_layers': best_num_layers,
    'activation_function': best_activation_function,
    'dropout_rate': best_dropout_rate
}

#-# torch.save saves the dictionary with model information:
#   - model information;
#   - model state dictionary;
#   - optimizer state dictionary 
torch.save({
    'model_info': model_info,
    'state_dict': best_model.state_dict(),
    'optimizer_state_dict': best_optimizer.state_dict(),
}, model_path)

print(f"Saved model to {model_path}")

#-# Predict values on the test data
test_losses = []
test_predictions = []
original_targets = []
predicted_indices = []

for idx, data in zip(test_indices, test_graphs):
    
    #-# Set the model to evaluation mode to disable dropout and batch normalization
    best_model.eval()

    #-# Disable gradient computation to speed up calculations and save memory
    with torch.no_grad():
        
        #-# Obtain model predictions for the current graph data
        output = best_model(data)

        #-# Compute the loss function on the current test data
        test_loss = criterion(output, data.y).item()
        test_losses.append(test_loss)
        
        #-# Inverse transform model predictions from normalized state
        model_output_unscaled = target_scaler.inverse_transform(output.cpu().numpy().reshape(1, -1)).flatten()
        
        #-# Append transformed predictions to test_predictions list
        test_predictions.append(model_output_unscaled)
        
        #-# Inverse transform the original target values from normalized state
        original_target_unscaled = target_scaler.inverse_transform(data.y.cpu().numpy().reshape(1, -1)).flatten()
        original_targets.append(original_target_unscaled)
        
        #-# Append the index of the current graph to the predicted_indices list
        predicted_indices.append(idx)

#-# Compute the average loss function on the test dataset
average_test_loss = np.mean(test_losses)
print(f"Average Test Loss of Scaled Data: {average_test_loss}")

#-# Compute the mean absolute error between concatenated arrays of original target values and predictions
mae = mean_absolute_error(np.concatenate(original_targets), np.concatenate(test_predictions))
print(f"Average Mean Absolute Error (MAE) of Inversed Data: {mae}")

#-# Print indices of predicted graphs
print(f"Indices of Predicted Graphs: {predicted_indices}")


#----------# Visualization #----------#


original_targets = np.array(original_targets)
test_predictions = np.array(test_predictions)

#-# Set title
num_targets = original_targets.shape[1]
target_names = [
    'Gibbs Energy', 'Electronic Energy', 'Entropy', 
    'Enthalpy', 'Dipole Moment', 'Band Gap'
]

#-# Create a .csv file with three columns: index, actual value, and predicted value
data = {
    'Index': predicted_indices,
    'Actual_Gibbs_Energy': original_targets[:, 0],
    'Predicted_Gibbs_Energy': test_predictions[:, 0],
    'Actual_Electronic_Energy': original_targets[:, 1],
    'Predicted_Electronic_Energy': test_predictions[:, 1],
    'Actual_Entropy': original_targets[:, 2],
    'Predicted_Entropy': test_predictions[:, 2],
    'Actual_Enthalpy': original_targets[:, 3],
    'Predicted_Enthalpy': test_predictions[:, 3],
    'Actual_Dipole_Moment': original_targets[:, 4],
    'Predicted_Dipole_Moment': test_predictions[:, 4],
    'Actual_Band_Gap': original_targets[:, 5],
    'Predicted_Band_Gap': test_predictions[:, 5]
}

df = pd.DataFrame(data)

#-# Save the file
csv_path = "./test_results.csv"
df.to_csv(csv_path, index=False)
print(f"Saved DataFrame to {csv_path}")

#-# Calculate metrics
maes = []
mapes = []
mse_values = []
rmse_values = []
r2_values = []

for i in range(num_targets):
    mae = mean_absolute_error(original_targets[:, i], test_predictions[:, i])
    maes.append(mae)
    
    mape = mean_absolute_percentage_error(original_targets[:, i], test_predictions[:, i])
    mapes.append(mape)
    
    mse = mean_squared_error(original_targets[:, i], test_predictions[:, i])
    mse_values.append(mse)
    
    rmse = np.sqrt(mse)
    rmse_values.append(rmse)
    
    r2 = r2_score(original_targets[:, i], test_predictions[:, i])
    r2_values.append(r2)

#-# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
axes = axes.flatten()

#-# Create scatter plot with conditional coloring (red if observation number < 48)
for i in range(num_targets):
    ax = axes[i]
    for idx in range(len(df)):
        color = 'red' if df['Index'].iloc[idx] < 48 else 'blue'
        ax.scatter(df[f'Actual_{target_names[i].replace(" ", "_")}'].iloc[idx],
                   df[f'Predicted_{target_names[i].replace(" ", "_")}'].iloc[idx],
                   alpha=0.5, color=color)
    
    #-# Create diagonal line
    ax.plot([original_targets[:, i].min(), original_targets[:, i].max()],
            [original_targets[:, i].min(), original_targets[:, i].max()],
            color='black', linestyle='--')
    
    #-# Set titles for axes 
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(target_names[i])
    ax.grid(True)
    
    #-# Add metrics to the plots
    ax.text(0.05, 0.95, f"MSE: {mse_values[i]:.4f}\nRMSE: {rmse_values[i]:.4f}\nR-squared: {r2_values[i]:.4f}\nMAE: {maes[i]:.4f}\nMAPE: {mapes[i]:.2f}%", 
            transform=ax.transAxes, fontsize=12, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

for j in range(num_targets, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plot_path = "./test_results.png"
plt.savefig(plot_path)
print(f"Saved plots to {plot_path}")