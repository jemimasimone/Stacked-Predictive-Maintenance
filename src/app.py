import streamlit as st
import requests
import json
import os

SCORING_URI = "https://pdm-stacked-model.southeastasia.inference.ml.azure.com/score"
API_KEY = "6fqbOYAIllFUZTJkPHBWJ0SbeS4JwwSWTcqygx40NctMVXOk9N95JQQJ99BEAAAAAAAAAAAAINFRAZML20cs"

st.set_page_config(page_title="Predictive Maintenance Demo", layout="centered")

st.title("⚙️ Predictive Maintenance Model Demo")
st.markdown("Enter machine sensor readings to predict potential equipment failure.")

st.sidebar.header("Input Parameters")

# Input fields using Streamlit widgets
# You can use st.sidebar for inputs to make the main area cleaner
machine_type = st.sidebar.selectbox("Machine type", ["L", "M", "H"])
air_temp = st.sidebar.number_input("Air temperature (°C)", value=298.5, format="%.1f")
process_temp = st.sidebar.number_input("Process temperature (°C)", value=310.2, format="%.1f")
rotational_speed = st.sidebar.number_input("Rotational speed (rpm)", value=2861.0, format="%.1f")
torque = st.sidebar.number_input("Torque (Nm)", value=4.0, format="%.1f")
tool_wear = st.sidebar.number_input("Tool wear (min)", value=143.0, format="%.1f")

st.write("---") # Separator for visual appeal

if st.button("🚀 Predict Failure"):
    # Prepare the input data in the format your model expects
    input_data = {
        "Machine type": machine_type,
        "Air temperature": air_temp,
        "Process temperature": process_temp,
        "Rotational speed": rotational_speed,
        "Torque": torque,
        "Tool wear": tool_wear
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    st.subheader("Prediction Result:")
    try:
        # Make the POST request to your Azure ML endpoint
        response = requests.post(SCORING_URI, data=json.dumps(input_data), headers=headers)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

        prediction_result = response.json()

        if prediction_result.get("failure") == "yes":
            st.error(f"⚠️ **FAILURE PREDICTED!**")
            st.write(f"**Potential Failure Type:** **`{prediction_result.get('failure_type', 'Unknown')}`**")
            st.warning("Immediate attention or scheduled maintenance is recommended.")
        else:
            st.success("✅ **No Failure Predicted.**")
            st.write("Machine appears to be operating within normal parameters.")
            st.info("Continue regular monitoring.")

    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while calling the model endpoint: {e}")
        if hasattr(response, 'status_code'):
            st.error(f"Status Code: {response.status_code}")
            st.error(f"Response Text: {response.text}")
    except json.JSONDecodeError:
        st.error("Failed to decode JSON response from the model.")
        st.error(f"Raw response: {response.text}")

st.write("---")
st.caption(f"Powered by Azure ML Endpoint: {SCORING_URI}")
