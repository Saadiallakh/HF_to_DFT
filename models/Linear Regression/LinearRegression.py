import pandas as pd 
import numpy as np
import os

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

import matplotlib.pyplot as plt
import seaborn as sns
import joblib


#----------# Data Loading #----------#


data = pd.read_csv("./NN_ML.csv",delimiter=",",encoding="utf-8")
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
        lines = file.readlines()[2:]           
        coordinates = []
        for line in lines:
            parts = line.split()
            if len(parts) == 4:              
                coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return np.array(coordinates)

for index, row in data.iterrows():
    file_name = str(row['name']) + ".xyz"
    file_path = os.path.join("./xyz", file_name)
    coordinates = read_xyz_file(file_path)
    coordinates_list.append(coordinates)

data['coordinates'] = coordinates_list


#----------# Normalization and Dataset Splitting #----------#


features = ['hf_gibbs_free_energy_ev', 'hf_electronic_energy_ev', 'hf_entropy_ev',
            'hf_enthalpy_ev', 'hf_dipole_moment_d', 'hf_gap_ev', 'coordinates']

targets = ['dft_gibbs_free_energy_ev', 'dft_electronic_energy_ev', 'dft_entropy_ev',
           'dft_enthalpy_ev', 'dft_dipole_moment_d', 'dft_gap_ev']

X = data[features].copy()
y = data[targets].copy()

X['coordinates'] = X['coordinates'].apply(np.ravel)
max_length = X['coordinates'].apply(len).max()

X['coordinates'] = X['coordinates'].apply(lambda x: np.pad(x, (0, max_length - len(x)), mode='constant'))

X_numeric = pd.concat([X.drop(columns='coordinates'),
                       pd.DataFrame(np.vstack(X['coordinates']), 
                                    columns=[f'coord_{i}' for i in range(max_length)], 
                                    index=X.index)],
                      axis=1)

X_train_numeric, X_test_numeric, y_train, y_test = train_test_split(X_numeric, y, test_size=0.1, random_state=42)
test_indices_list = X_test_numeric.index.tolist()

#-# Normalization
scaler = StandardScaler()
all_columns = X_train_numeric.columns
X_train_numeric_scaled = scaler.fit_transform(X_train_numeric[all_columns])
X_test_numeric_scaled = scaler.transform(X_test_numeric[all_columns])
X_train_numeric_scaled = pd.DataFrame(X_train_numeric_scaled, columns=all_columns, index=X_train_numeric.index)
X_test_numeric_scaled = pd.DataFrame(X_test_numeric_scaled, columns=all_columns, index=X_test_numeric.index)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

scaler_dir = './output'
os.makedirs(scaler_dir, exist_ok=True)
joblib.dump(scaler, os.path.join(scaler_dir, 'scaler.pkl'))
joblib.dump(scaler_y, os.path.join(scaler_dir, 'scaler_y.pkl'))


#----------# Linear Regression #----------#


model = LinearRegression()
model.fit(X_train_numeric_scaled, y_train_scaled)
model_dir = "./output"

joblib.dump(model, os.path.join(model_dir, 'linear_regression_model.pkl'))
print(f"Trained model saved to {os.path.join(model_dir, 'linear_regression_model.pkl')}")
 
#-# Predict
y_pred_scaled = model.predict(X_test_numeric_scaled)
y_pred_inverse = scaler_y.inverse_transform(y_pred_scaled)
y_test_inverse = scaler_y.inverse_transform(y_test_scaled) 


#----------# Visualization #----------#


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
df['Color'] = df['Index'].apply(lambda x: 'red' if x < 48 else 'blue')

csv_path = "./test_results.csv"
df.to_csv(csv_path, index=False)
print(f"Saved DataFrame to {csv_path}")

target_names = [
    'Gibbs Energy', 'Electronic Energy', 'Entropy', 
    'Enthalpy', 'Dipole Moment', 'Band Gap'
]

#-# Metrics
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

fig, axs = plt.subplots(2, 3, figsize=(18, 12))
axs = axs.flatten()

for i, target in enumerate(target_names):
    ax = axs[i]
    
    sns.scatterplot(
        x=df[f'Actual_{target.replace(" ", "_")}'],
        y=df[f'Predicted_{target.replace(" ", "_")}'],
        hue=df['Color'],
        palette={'red': 'red', 'blue': 'blue'},
        ax=ax,
        alpha=0.5,
        legend=False
    )

    ax.plot(
        [df[f'Actual_{target.replace(" ", "_")}'].min(), df[f'Actual_{target.replace(" ", "_")}'].max()],
        [df[f'Actual_{target.replace(" ", "_")}'].min(), df[f'Actual_{target.replace(" ", "_")}'].max()],
        color='black', linestyle='--'
    )
    
    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title(target, fontsize=14)
    ax.grid(True)
    
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
