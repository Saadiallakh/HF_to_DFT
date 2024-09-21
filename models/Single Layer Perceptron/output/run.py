import streamlit as st
import numpy as np
import joblib
import pandas as pd

from tensorflow.keras.models import load_model

#-# Load the pre-trained model and scalers
model = load_model("slp_model.keras")
scaler = joblib.load("scaler.pkl")
scaler_y = joblib.load("scaler_y.pkl")

#-# Define the maximum length for padded coordinates
max_length = 216

#-# Configure Streamlit page
st.set_page_config(page_title="Single Layer Perceptron", page_icon=":bar_chart:")

#-# Initialize session state variables if not already present
if 'page' not in st.session_state:
    st.session_state['page'] = 'input'

if 'input_data' not in st.session_state:
    st.session_state['input_data'] = None

def show_input_page():
    """
    Displays the input page for the Single Layer Perceptron model.
    Allows users to input required values and upload an XYZ file.
    """
    #-# Display the title and description
    st.markdown("<h1 style='text-align: center;'>Single Layer Perceptron</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: grey; font-size: 24px;'>Enter values calculated using 'HF-3c D4' approximation</p>", unsafe_allow_html=True)

    #-# Apply custom styling for input fields
    st.markdown("""
        <style>
            .input-label { font-weight: bold; margin-bottom: -50px; }
            .input-field { margin-top: 0px; font-size: 20px; } /* Apply 20px font size to input fields */
            .css-1x0t1ku { font-size: 20px; } /* Adjust font size of number input fields */
        </style>
    """, unsafe_allow_html=True)

    #-# Define columns for input fields
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <p class='input-label'>Gibbs Free Energy (eV) <span style='color: red;'>*</span></p>
            """, unsafe_allow_html=True)
        hf_gibbs_free_energy_ev = st.number_input(
            "", format="%.4f", value=0.0, step=0.0001, key="gibbs_free_energy", label_visibility="hidden"
        )
        
        st.markdown("""
            <p class='input-label'>Electronic Energy (eV) <span style='color: red;'>*</span></p>
            """, unsafe_allow_html=True)
        hf_electronic_energy_ev = st.number_input(
            "", format="%.4f", value=0.0, step=0.0001, key="electronic_energy", label_visibility="hidden"
        )
        
        st.markdown("""
            <p class='input-label'>Entropy (eV) <span style='color: red;'>*</span></p>
            """, unsafe_allow_html=True)
        hf_entropy_ev = st.number_input(
            "", format="%.4f", value=0.0, step=0.0001, key="entropy", label_visibility="hidden"
        )

    with col2:
        st.markdown("""
            <p class='input-label'>Enthalpy (eV) <span style='color: red;'>*</span></p>
            """, unsafe_allow_html=True)
        hf_enthalpy_ev = st.number_input(
            "", format="%.4f", value=0.0, step=0.0001, key="enthalpy", label_visibility="hidden"
        )
        
        st.markdown("""
            <p class='input-label'>Dipole Moment (D) <span style='color: red;'>*</span></p>
            """, unsafe_allow_html=True)
        hf_dipole_moment_d = st.number_input(
            "", format="%.4f", value=0.0, step=0.0001, key="dipole_moment", label_visibility="hidden"
        )
        
        st.markdown("""
            <p class='input-label'>Band Gap (eV) <span style='color: red;'>*</span></p>
            """, unsafe_allow_html=True)
        hf_gap_ev = st.number_input(
            "", format="%.4f", value=0.0, step=0.0001, key="band_gap", label_visibility="hidden"
        )

    #-# Input for uploading XYZ file
    st.markdown("""
        <p class='input-label' style='font-weight: bold;'>Upload XYZ file <span style='color: red;'>*</span></p>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type="xyz", label_visibility="hidden")

    def read_xyz_file(file):
        """
        Reads XYZ file and extracts atomic coordinates.

        Parameters:
            file: Uploaded XYZ file.

        Returns:
            numpy.ndarray: Array of atomic coordinates.
        """
        lines = file.readlines()[2:]
        coordinates = []
        for line in lines:
            parts = line.split()
            if len(parts) == 4:
                coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return np.array(coordinates)

    if uploaded_file is not None:
        coordinates = read_xyz_file(uploaded_file).ravel()
        padded_coordinates = np.pad(coordinates, (0, max_length - len(coordinates)), mode='constant')
        st.session_state['input_data'] = np.array([[hf_gibbs_free_energy_ev, hf_electronic_energy_ev, hf_entropy_ev,
                                                    hf_enthalpy_ev, hf_dipole_moment_d, hf_gap_ev, *padded_coordinates]])
        st.session_state['file_uploaded'] = True
    else:
        st.session_state['file_uploaded'] = False

    def is_filled(value):
        """
        Checks if an input value is filled (non-zero).

        Parameters:
            value: Input value.

        Returns:
            bool: True if the value is non-zero, False otherwise.
        """
        return value != 0.0

    #-# Check if all required fields are filled
    all_inputs_provided = (
        is_filled(hf_gibbs_free_energy_ev) and 
        is_filled(hf_electronic_energy_ev) and 
        is_filled(hf_entropy_ev) and 
        is_filled(hf_enthalpy_ev) and 
        is_filled(hf_dipole_moment_d) and 
        is_filled(hf_gap_ev) and 
        st.session_state['file_uploaded']
    )

    #-# Display status message based on input completeness
    if all_inputs_provided:
        st.markdown("<p style='color: green;'>✓ All mandatory fields filled</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: red;'>* Mandatory fields</p>", unsafe_allow_html=True)

    #-# Button to trigger prediction
    if st.button("Predict", disabled=not all_inputs_provided):
        st.session_state['page'] = 'results'
    
    st.markdown("<p style='text-align: left; color: grey;'>Double click the button to predict</p>", unsafe_allow_html=True)

def show_results_page():
    """
    Displays the results page with the predicted DFT properties.
    Shows predictions in a table format and provides an option to restart.
    """    
    if 'input_data' in st.session_state and st.session_state['input_data'] is not None:
        
        #-# Scale the input data and make predictions
        input_data_scaled = scaler.transform(st.session_state['input_data'])
        prediction_scaled = model.predict(input_data_scaled)
        prediction = scaler_y.inverse_transform(prediction_scaled.reshape(1, -1))

        st.markdown("<h1 style='text-align: center;'>Predicted DFT Properties</h1>", unsafe_allow_html=True)

        #-# Prepare results for display
        results_df = pd.DataFrame({
            'Property': ['Gibbs Free Energy (eV)', 'Electronic Energy (eV)', 'Entropy (eV)', 'Enthalpy (eV)', 'Dipole Moment (D)', 'Band Gap (eV)'],
            'Prediction': [f"{prediction[0][0]:.4f}", f"{prediction[0][1]:.4f}", f"{prediction[0][2]:.4f}", f"{prediction[0][3]:.4f}", f"{prediction[0][4]:.4f}", f"{prediction[0][5]:.4f}"]
        })

        #-# Generate HTML table for displaying the predictions
        table_html = results_df.reset_index().rename(columns={'index': 'No.'}).to_html(index=False, border=0, classes='custom-table')

        #-# Apply custom styles to the table
        st.markdown("""
            <style>
                .custom-table {
                    width: 100%;
                    border-collapse: collapse;
                }
                .custom-table th {
                    background-color: black;
                    color: white;
                    font-weight: bold;
                    text-align: center;
                    padding: 8px;
                }
                .custom-table td {
                    padding: 8px;
                    text-align: center;
                }
                .custom-table tr:nth-child(even) {
                    background-color: #f2f2f2;
                }
                .custom-table tr:nth-child(odd) {
                    background-color: white;
                }
                .custom-table .no-column {
                    color: black;
                    font-weight: bold;
                    text-align: center;
                }
            </style>
        """, unsafe_allow_html=True)

        st.markdown(table_html, unsafe_allow_html=True)

        #-# Button to restart the process
        if st.button("Restart"):
            st.session_state['page'] = 'input'
        
        st.markdown("<p style='text-align: left; color: grey;'>Double click the button to restart</p>", unsafe_allow_html=True)

#-# Determine which page to show based on the session state
if st.session_state['page'] == 'input':
    show_input_page()
elif st.session_state['page'] == 'results':
    show_results_page()