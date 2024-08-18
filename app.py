import streamlit as st
from fastapi import FastAPI
from fastapi.responses import FileResponse
import threading
import uvicorn
import os

# Initialize the FastAPI app
api = FastAPI()

# Initialize Streamlit UI
def run_streamlit():
    st.title("Criteria Data API")
    st.write("This app serves CSV files via an API endpoint using FastAPI.")

    # Instructions for the user
    st.write("### API Endpoints")
    st.write("You can access the CSV data for different criteria using the following URLs:")
    st.code("http://localhost:8000/api/economic_viability", language="python")
    st.code("http://localhost:8000/api/reliability", language="python")
    st.code("http://localhost:8000/api/scalability", language="python")
    st.code("http://localhost:8000/api/infrastructure_integration", language="python")

# Define FastAPI routes
@api.get("/api/{criterion}")
async def get_csv(criterion: str):
    files = {
        "economic_viability": "economic_viability_dynamics.csv",
        "reliability": "reliability_curve_dynamics.csv",
        "scalability": "scalability_curve_dynamics.csv",
        "infrastructure_integration": "infrastructure_integration_dynamics.csv",
        # Add other criteria files here
    }
    
    if criterion in files:
        return FileResponse(files[criterion], media_type='text/csv', filename=f"{criterion}.csv")
    else:
        return {"error": "Criterion not found"}

# Run FastAPI in a separate thread
def run_fastapi():
    uvicorn.run(api, host="0.0.0.0", port=8000)

# Start the FastAPI and Streamlit servers
if __name__ == "__main__":
    threading.Thread(target=run_fastapi, daemon=True).start()
    run_streamlit()