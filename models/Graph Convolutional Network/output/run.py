import streamlit as st
import numpy as np
import joblib
import tempfile
import os

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

#-# Check if the GPU is available on the device
#   If not available, the process will run on CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#-# Set up Streamlit page configuration
st.set_page_config(page_title="Graph Convolutional Network", page_icon=":bar_chart:")

#-# Initialize session state if not already present
if 'page' not in st.session_state:
    st.session_state['page'] = 'input'

if 'input_data' not in st.session_state:
    st.session_state['input_data'] = None

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
    morgan_fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=256)
    return np.array(morgan_fp)

def read_sdf_file(file_path):
    """
    Read molecules from SDF file.

    Parameters:
        file_path (str): Path to the SDF file.

    Returns:
        list: List of RDKit molecule objects.
    """
    molecules = []
    suppl = Chem.SDMolSupplier(file_path)
    for mol in suppl:
        if mol is not None:
            sanitized_mol = sanitize_molecule(mol)
            if sanitized_mol is not None:
                molecules.append(sanitized_mol)
    return molecules

def create_graph_for_single_system(hf_features, sdf_file, feature_scaler):
    """
    Create a graph representation for a single system from HF features and an SDF file.

    Parameters:
        hf_features (np.ndarray): Array of HF features.
        sdf_file (UploadedFile): Uploaded SDF file.
        feature_scaler (StandardScaler): Scaler for feature normalization.

    Returns:
        Data: PyTorch Geometric Data object containing node features and edge indices.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.sdf') as tmp_file:
        tmp_file.write(sdf_file.read())
        tmp_file.flush()
        sdf_file_path = tmp_file.name
    
    molecules = read_sdf_file(sdf_file_path)
    os.remove(sdf_file_path) 
    
    if not molecules:
        raise ValueError("No valid molecules found in the uploaded file.")
    
    sanitized_mol = molecules[0]
    rdkit_features = compute_rdkit_features(sanitized_mol)
    sanitized_mol = Chem.AddHs(sanitized_mol)
    num_atoms = sanitized_mol.GetNumAtoms()
    
    #-# Generate node features
    node_features_list = []
    for atom_idx in range(num_atoms):
        atom = sanitized_mol.GetAtomWithIdx(atom_idx)
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
    
    #-# Generate edge indices
    edge_index = []
    for bond in sanitized_mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    
    graph_data = Data(x=node_features, edge_index=edge_index)
    
    #-# Normalize features
    graph_data.x = torch.tensor(feature_scaler.transform(graph_data.x.cpu().numpy()), dtype=torch.float64).to(device)
    
    return graph_data

#-# Load model checkpoint
checkpoint = torch.load("best_gcn_model.pt")
model_info = checkpoint['model_info']
state_dict = checkpoint['state_dict']

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

#-# Instantiate and load the best model
best_model = GCN(
    input_dim=model_info['input_dim'],
    hidden_dim=model_info['hidden_dim'],
    output_dim=model_info['output_dim'],
    num_layers=model_info['num_layers'],
    activation_function=model_info['activation_function'],
    dropout_rate=model_info['dropout_rate']
).double()

best_model.load_state_dict(state_dict)
best_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

#-# Load scalers
feature_scaler = joblib.load("feature_scaler.pkl")
target_scaler = joblib.load("target_scaler.pkl")

def show_input_page():
    """
    Display the input page for user to enter HF features and upload an SDF file.
    """
    st.markdown("<h1 style='text-align: center;'>Graph Convolutional Network</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: grey; font-size: 24px;'>Enter values calculated using 'HF-3c D4' approximation</p>", unsafe_allow_html=True)

    st.markdown("""
        <style>
            .input-label { font-weight: bold; margin-bottom: -50px; }
            .input-field { margin-top: 0px; font-size: 20px; }
            .css-1x0t1ku { font-size: 20px; }
        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<p class='input-label'>Gibbs Free Energy (eV) <span style='color: red;'>*</span></p>", unsafe_allow_html=True)
        gibbs_energy = st.number_input("", format="%.4f", value=0.0, step=0.0001, key="gibbs_energy", label_visibility="hidden")
        
        st.markdown("<p class='input-label'>Electronic Energy (eV) <span style='color: red;'>*</span></p>", unsafe_allow_html=True)
        electronic_energy = st.number_input("", format="%.4f", value=0.0, step=0.0001, key="electronic_energy", label_visibility="hidden")
        
        st.markdown("<p class='input-label'>Entropy (eV) <span style='color: red;'>*</span></p>", unsafe_allow_html=True)
        entropy = st.number_input("", format="%.4f", value=0.0, step=0.0001, key="entropy", label_visibility="hidden")

    with col2:
        st.markdown("<p class='input-label'>Enthalpy (eV) <span style='color: red;'>*</span></p>", unsafe_allow_html=True)
        enthalpy = st.number_input("", format="%.4f", value=0.0, step=0.0001, key="enthalpy", label_visibility="hidden")
        
        st.markdown("<p class='input-label'>Dipole Moment (D) <span style='color: red;'>*</span></p>", unsafe_allow_html=True)
        dipole_moment = st.number_input("", format="%.4f", value=0.0, step=0.0001, key="dipole_moment", label_visibility="hidden")
        
        st.markdown("<p class='input-label'>Band Gap (eV) <span style='color: red;'>*</span></p>", unsafe_allow_html=True)
        homo_lumo_gap = st.number_input("", format="%.4f", value=0.0, step=0.0001, key="band_gap", label_visibility="hidden")

    st.markdown("<p class='input-label' style='font-weight: bold;'>Upload SDF file <span style='color: red;'>*</span></p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type="sdf", label_visibility="hidden")

    def is_filled(value):
        return value != 0.0

    all_inputs_provided = (
        is_filled(gibbs_energy) and 
        is_filled(electronic_energy) and 
        is_filled(entropy) and 
        is_filled(enthalpy) and 
        is_filled(dipole_moment) and 
        is_filled(homo_lumo_gap) and 
        uploaded_file is not None
    )

    if all_inputs_provided:
        st.markdown("<p style='color: green;'>✓ All mandatory fields filled</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: red;'>* Mandatory fields</p>", unsafe_allow_html=True)

    if st.button("Predict", disabled=not all_inputs_provided):
        st.session_state['page'] = 'results'
        try:
            hf_features = np.array([gibbs_energy, electronic_energy, entropy, enthalpy, dipole_moment, homo_lumo_gap])
            st.session_state['input_data'] = (hf_features, uploaded_file)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    st.markdown("<p style='text-align: left; color: grey;'>Double click the button to predict</p>", unsafe_allow_html=True)

def show_results_page():
    """
    Display the results page with predicted properties based on the input data.
    """
    if 'input_data' not in st.session_state or st.session_state['input_data'] is None:
        st.error("No input data found. Please go back to the input page and provide the necessary information.")
        return
    
    hf_features, sdf_file = st.session_state['input_data']
    
    try:
        graph_data = create_graph_for_single_system(hf_features, sdf_file, feature_scaler)

        best_model.eval()
        with torch.no_grad():
            output = best_model(graph_data)
            predicted_properties = target_scaler.inverse_transform(output.cpu().numpy().reshape(1, -1))

        st.markdown("<h1 style='text-align: center;'>Predicted DFT Properties</h1>", unsafe_allow_html=True)

        results_df = pd.DataFrame({
            'Property': ['Gibbs Free Energy (eV)', 'Electronic Energy (eV)', 'Entropy (eV)', 'Enthalpy (eV)', 'Dipole Moment (D)', 'Band Gap (eV)'],
            'Prediction': [f"{pred:.4f}" for pred in predicted_properties[0]]
        })

        table_html = results_df.reset_index().rename(columns={'index': 'No.'}).to_html(index=False, border=0, classes='custom-table')
        
        st.markdown("""
        <style>
            .custom-table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px auto;
            }
            .custom-table th {
                background-color: black;
                color: white;
                font-weight: bold;
                text-align: center;
                padding: 10px;
            }
            .custom-table td {
                padding: 10px;
                text-align: center;
            }
            .custom-table tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            .custom-table tr:nth-child(odd) {
                background-color: #ffffff;
            }
        </style>
        """, unsafe_allow_html=True)

        st.markdown(table_html, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

    if st.button("Restart"):
        st.session_state['page'] = 'input'
        st.session_state['input_data'] = None
        st.markdown(
            "<meta http-equiv='refresh' content='0'>",
            unsafe_allow_html=True
        )

#-# Render the appropriate page based on session state
if st.session_state['page'] == 'input':
    show_input_page()
elif st.session_state['page'] == 'results':
    show_results_page()