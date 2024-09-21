import pandas as pd 
import numpy as np
import os
import tensorflow as tf
import optuna

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1_l2

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

import matplotlib.pyplot as plt
import seaborn as sns
import joblib


#----------# Data Loading #----------#


#-# Load data from a CSV file
data = pd.read_csv("./NN_ML.csv",delimiter=",",encoding="utf-8")

#-# Create an empty list to store the '.xyz' coordinates
coordinates_list = []

def read_xyz_file(file_path):
    """
    Function to read XYZ format files and extract atomic coordinates.

    Parameters:
    - file_path (str): Path to the XYZ file.

    Returns:
    - np.array: Numpy array of atomic coordinates.
    """
    with open(file_path, 'r') as file:

        #-# Skip the first two lines containing metadata
        lines = file.readlines()[2:]           
        coordinates = []
        for line in lines:
            parts = line.split()

            #-# Check if the line contains coordinates
            if len(parts) == 4:              
                coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return np.array(coordinates)

#-# For each row in the DataFrame, form the path to the file and extract coordinates
for index, row in data.iterrows():
    file_name = str(row['name']) + ".xyz"
    file_path = os.path.join("./xyz", file_name)
    coordinates = read_xyz_file(file_path)
    coordinates_list.append(coordinates)

#-# Add the coordinates to the DataFrame as a new column
data['coordinates'] = coordinates_list


#----------# Normalization and Dataset Splitting #----------#


#-# Define feature and target sets
features = ['hf_gibbs_free_energy_ev', 'hf_electronic_energy_ev', 'hf_entropy_ev',
            'hf_enthalpy_ev', 'hf_dipole_moment_d', 'hf_gap_ev', 'coordinates']

targets = ['dft_gibbs_free_energy_ev', 'dft_electronic_energy_ev', 'dft_entropy_ev',
           'dft_enthalpy_ev', 'dft_dipole_moment_d', 'dft_gap_ev']

#-# Create DataFrame for features and target variables
X = data[features].copy()
y = data[targets].copy()

#-# Flatten the coordinates into 1D arrays
X['coordinates'] = X['coordinates'].apply(np.ravel)

#-# Find the maximum length of the coordinate arrays
max_length = X['coordinates'].apply(len).max()

#-# Pad coordinates with zeros to the maximum length
X['coordinates'] = X['coordinates'].apply(lambda x: np.pad(x, (0, max_length - len(x)), mode='constant'))

#-# Create a new DataFrame with coordinates as individual columns
X_numeric = pd.concat([X.drop(columns='coordinates'),
                       pd.DataFrame(np.vstack(X['coordinates']), 
                                    columns=[f'coord_{i}' for i in range(max_length)], 
                                    index=X.index)],
                      axis=1)

#-# Split data into training and test sets
X_train_numeric, X_test_numeric, y_train, y_test = train_test_split(X_numeric, y, test_size=0.1, random_state=42)

#-# Track indices of the test set
test_indices_list = X_test_numeric.index.tolist()

#-# Normalize the training data
scaler = StandardScaler()

#-# Save column names for later DataFrame reconstruction
all_columns = X_train_numeric.columns

#-# Apply standardization to the training data
X_train_numeric_scaled = scaler.fit_transform(X_train_numeric[all_columns])

#-# Apply the same transformation to the test data
X_test_numeric_scaled = scaler.transform(X_test_numeric[all_columns])

#-# Reconstruct DataFrame with normalized data for the training set
X_train_numeric_scaled = pd.DataFrame(X_train_numeric_scaled, columns=all_columns, index=X_train_numeric.index)

#-# Reconstruct DataFrame with normalized data for the test set
X_test_numeric_scaled = pd.DataFrame(X_test_numeric_scaled, columns=all_columns, index=X_test_numeric.index)

#-# Normalize the target values
scaler_y = StandardScaler()

#-# Apply standardization to the target variables in the training set
y_train_scaled = scaler_y.fit_transform(y_train)

#-# Apply the same transformation to the target variables in the test set
y_test_scaled = scaler_y.transform(y_test)

#-# Save the scalers
scaler_dir = './output'
os.makedirs(scaler_dir, exist_ok=True)

joblib.dump(scaler, os.path.join(scaler_dir, 'scaler.pkl'))
joblib.dump(scaler_y, os.path.join(scaler_dir, 'scaler_y.pkl'))


#----------# Single Layer Perceptron #----------#


#-# Check if the GPU is available on the device
if tf.config.list_physical_devices('GPU'):
    print("GPU is available. Running on GPU.")
else:
    print("GPU is not available. Running on CPU.")

#-# If detected, then permit growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

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
    optimizer_name = trial.suggest_categorical('optimizer', ['Ftrl', 'Adadelta', 'Lion', 'Adamax', 'Adam', 'SGD', 'RMSprop', 'Adagrad', 'AdamW'])
    activation_function = trial.suggest_categorical('activation_function', ['relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu', 'gelu', 'silu', 'mish'])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    num_neurons = trial.suggest_int('num_neurons', 5, 200, log=True)
    l1_reg = trial.suggest_float('l1_reg', 1e-8, 100, log=True)
    l2_reg = trial.suggest_float('l2_reg', 1e-8, 100, log=True)
    
    #-# Create a model with the current hyperparameter values
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_numeric_scaled.shape[1],)),
        tf.keras.layers.Dense(num_neurons, activation=activation_function, kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),
        tf.keras.layers.Dense(6)
    ])
    
    #-# Define optimizer 
    if optimizer_name == 'Ftrl':
        optimizer = tf.keras.optimizers.Ftrl(learning_rate=learning_rate)
    elif optimizer_name == 'Adadelta':
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
    elif optimizer_name == 'Lion':
        optimizer = tf.keras.optimizers.Lion(learning_rate=learning_rate)
    elif optimizer_name == 'Adamax':
        optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
    elif optimizer_name == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'RMSprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'Adagrad':
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer_name == 'AdamW':
        optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
    elif optimizer_name == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError("Invalid optimizer specified")
    
    #-# Compile the model with optimizer
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
    
    #-# Create a KFold object for 3-fold cross-validation with data shuffling
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    fold_mses = []

    #-# Loop through the data splits
    for train_index, val_index in kf.split(X_train_numeric_scaled):
        X_train_cv, X_val_cv = X_train_numeric_scaled.iloc[train_index], X_train_numeric_scaled.iloc[val_index]
        y_train_cv, y_val_cv = y_train_scaled[train_index], y_train_scaled[val_index]
        
        #-# Apply EarlyTopping technique to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, patience=20, verbose=0)
        
        #-# Train the model
        history = model.fit(X_train_cv, y_train_cv, epochs=500, batch_size=16, validation_data=(X_val_cv, y_val_cv),
                            callbacks=[early_stopping], verbose=0)
        
        #-# Evaluate on the validation set
        y_pred_scaled = model.predict(X_val_cv)        
        mse = mean_squared_error(y_val_cv, y_pred_scaled)
        fold_mses.append(mse)
    
    return np.mean(fold_mses)

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
best_optimizer = best_params['optimizer']
best_activation_function = best_params['activation_function']
best_learning_rate = best_params['learning_rate']
best_neurons = best_params['num_neurons']
best_l1_reg = best_params['l1_reg']
best_l2_reg = best_params['l2_reg']

#-# Create the model with best hyperparameters 
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_numeric_scaled.shape[1],)),
    tf.keras.layers.Dense(best_neurons, activation=best_activation_function, kernel_regularizer=l1_l2(l1=best_l1_reg, l2=best_l2_reg)),
    tf.keras.layers.Dense(6)
])

#-# Get the best optimizer
if best_optimizer == 'Lion':
    optimizer = tf.keras.optimizers.Lion(learning_rate=best_learning_rate)
elif best_optimizer == 'Ftrl':
    optimizer = tf.keras.optimizers.Ftrl(learning_rate=best_learning_rate)
elif best_optimizer == 'Adamax':
    optimizer = tf.keras.optimizers.Adamax(learning_rate=best_learning_rate)
elif best_optimizer == 'Adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate=best_learning_rate)
elif best_optimizer == 'Adadelta':
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=best_learning_rate)
elif best_optimizer == 'RMSprop':
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=best_learning_rate)
elif best_optimizer == 'Adagrad':
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=best_learning_rate)
elif best_optimizer == 'AdamW':
    optimizer = tf.keras.optimizers.AdamW(learning_rate=best_learning_rate)
elif best_optimizer == 'SGD':
    optimizer = tf.keras.optimizers.SGD(learning_rate=best_learning_rate)
else:
    raise ValueError("Invalid optimizer specified")

#-# Apply EarlyTopping technique to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, patience=20, verbose=0)

#-# Compile the model with the best optimizer
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])

#-# Train the model using the best hyperparameters
history = model.fit(X_train_numeric_scaled, y_train_scaled, epochs=500, batch_size=16, validation_split=0.1, callbacks=[early_stopping], verbose=0)

#-# Plot Train & Val curves
#   Save them as training_validation_loss.png
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='brown')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.grid(True)

plot_filename = './training_validation_loss.png'
plt.savefig(plot_filename, bbox_inches='tight') 

print(f"Plot saved to {plot_filename}")

#-# Save the trained model
model_filename = './slp_model.keras'
model.save(model_filename)
print(f"Model saved to {model_filename}")

#-# Predict values on the test data
y_pred_scaled = model.predict(X_test_numeric_scaled)

#-# Inverse transform to get original values
y_pred_inverse = scaler_y.inverse_transform(y_pred_scaled)
y_test_inverse = scaler_y.inverse_transform(y_test_scaled) 


#----------# Visualization #----------#


#-# Create a .csv file with three columns: index, actual value, and predicted value
data = {
    'Index': test_indices_list,
    'Actual_Gibbs_Energy': y_test_inverse[:, 0],
    'Predicted_Gibbs_Energy': y_pred_inverse[:, 0],
    'Actual_Electronic_Energy': y_test_inverse[:, 1],
    'Predicted_Electronic_Energy': y_pred_inverse[:, 1],
    'Actual_Entropy': y_test_inverse[:, 2],
    'Predicted_Entropy': y_pred_inverse[:, 2],
    'Actual_Enthalpy': y_test_inverse[:, 3],
    'Predicted_Enthalpy': y_pred_inverse[:, 3],
    'Actual_Dipole_Moment': y_test_inverse[:, 4],
    'Predicted_Dipole_Moment': y_pred_inverse[:, 4],
    'Actual_Band_Gap': y_test_inverse[:, 5],
    'Predicted_Band_Gap': y_pred_inverse[:, 5]
}

df = pd.DataFrame(data)

#-# Add a column with the color of predicted points (red if observation number < 48)
df['Color'] = df['Index'].apply(lambda x: 'red' if x < 48 else 'blue')

#-# Save the file
csv_path = "./test_results.csv"
df.to_csv(csv_path, index=False)
print(f"Saved DataFrame to {csv_path}")

#-# Set title
target_names = [
    'Gibbs Energy', 'Electronic Energy', 'Entropy', 
    'Enthalpy', 'Dipole Moment', 'Band Gap'
]

#-# Calculate metrics
metrics = {}
for i, target in enumerate(target_names):
    mae = mean_absolute_error(y_test_inverse[:, i], y_pred_inverse[:, i])
    mape = mean_absolute_percentage_error(y_test_inverse[:, i], y_pred_inverse[:, i])
    mse = mean_squared_error(y_test_inverse[:, i], y_pred_inverse[:, i])
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_inverse[:, i], y_pred_inverse[:, i])
    
    metrics[target] = {
        'MSE': mse,
        'RMSE': rmse,
        'R-squared': r2,
        'MAE': mae,
        'MAPE': mape
    }

#-# Create subplots
fig, axs = plt.subplots(2, 3, figsize=(18, 12))
axs = axs.flatten()

for i, target in enumerate(target_names):
    ax = axs[i]
    
    #-# Create scatter plot with conditional coloring
    sns.scatterplot(
        x=df[f'Actual_{target.replace(" ", "_")}'],
        y=df[f'Predicted_{target.replace(" ", "_")}'],
        hue=df['Color'],
        palette={'red': 'red', 'blue': 'blue'},
        ax=ax,
        alpha=0.5,
        legend=False
    )
    
    #-# Create diagonal line
    ax.plot(
        [df[f'Actual_{target.replace(" ", "_")}'].min(), df[f'Actual_{target.replace(" ", "_")}'].max()],
        [df[f'Actual_{target.replace(" ", "_")}'].min(), df[f'Actual_{target.replace(" ", "_")}'].max()],
        color='black', linestyle='--'
    )
    
    #-# Set titles for axes 
    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title(target, fontsize=14)
    ax.grid(True)
    
    #-# Add metrics to the plots
    ax.text(
        0.05, 0.95,
        f"MSE: {metrics[target]['MSE']:.4f}\nRMSE: {metrics[target]['RMSE']:.4f}\nR-squared: {metrics[target]['R-squared']:.4f}\nMAE: {metrics[target]['MAE']:.4f}\nMAPE: {metrics[target]['MAPE']:.2f}%",
        transform=ax.transAxes, fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

for j in range(len(target_names), len(axs)):
    axs[j].axis('off')

plt.tight_layout()
plot_path = "./test_results.png"
plt.savefig(plot_path)
print(f"Saved plots to {plot_path}")