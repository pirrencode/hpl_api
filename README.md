# hpl_api

Welcome to the official repository of the **Hyperloop Decision Making Ecosystem (HDME)** project. This project is an integral part of Aleksejs Vesjolijs' PhD dissertation and aims to provide an advanced framework for decision-making in Hyperloop projects using ambient Artificial Intelligence (AI).

## Overview

The **Hyperloop Decision Making Ecosystem (HDME)** is a robust platform designed to optimize decision-making processes in Hyperloop systems. The repository contains APIs, simulations, and tools necessary for analyzing, designing, and implementing Hyperloop networks. It leverages state-of-the-art methodologies such as the E(G)TL Model and system dynamics modeling.

---

## HDME Research Publications

The HDME project has been featured in several peer-reviewed research publications, demonstrating its scientific and practical contributions:

1. **Vesjolijs, A.** (2024). *The E(G)TL Model: A Novel Approach for Efficient Data Handling and Extraction in Multivariate Systems*. Applied System Innovation, Switzerland, 7(5), p.92. [DOI: 10.3390/asi7050092](https://doi.org/10.3390/asi7050092)
2. **Vesjolijs, A.** (2024). *Hyperloop Decision Making Ecosystem Empowered by Ambient Artificial Intelligence*. Future Tech Conference 2024, London, United Kingdom.
3. **Vesjolijs, A.** (2024). *Hyperloop Routes Optimization Considering Barren Soil: Case of Latvia*. International Conference on Reliability and Statistics in Transportation and Communication 2024.
4. **Vesjolijs, A.** (2024). *Implementation Framework for Hyperloop Decision-Making Ecosystem*. Digital Baltic & Information Systems 2024 (DB&IS 2024). [CEUR: Paper 10](https://ceur-ws.org/Vol-3698/paper10.pdf)
5. **Vesjolijs, A.** (2024). *Overview of Factors and Methods for Analysis of Hyperloop Project*. Springer Nature Switzerland, Cham, pp. 281–291. [DOI: 10.1007/978-3-031-53598-7_25](https://doi.org/10.1007/978-3-031-53598-7_25)

---

## Features

1. **System Dynamics Dashboard**:
   - A dynamic visualization tool for analyzing Hyperloop system metrics and scenarios.
   - Built-in functionality for exploring various decision-making parameters.

2. **EGTL Model Integration**:
   - Implements the E(G)TL Model for data transformation and efficient decision-making.

3. **Simulation Scenarios**:
   - Predefined simulation scenarios for analyzing key metrics such as social acceptance (SAC), technological feasibility (TFE), and economic viability.

4. **Database Management**:
   - Structured table creation DDL scripts for EGTL stores:
     - `swf_fusion_store.sql`
     - `swf_staging_store.sql`
     - `swf_alliance_store.sql`

5. **Unit Testing Framework**:
   - Comprehensive unit tests for key components of the project.
   - Test cases located under `tests/unit`.

---

## Installation

### Prerequisites
1. Python 3.8+
2. Dependencies listed in `requirements.txt`.

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/pirrencode/hpl_api.git

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Configure your Snowflake connection settings in the application.

# Usage

## Running the Application
1. Start the dashboard:
    ```bash
    python app.py
2. Navigate to the provided URL to access the dashboard.

## Running Unit Tests
To execute unit tests, use:
    ```bash
    pytest tests/unit/{unit_test_name}.py

## Repository Structure

```plaintext
hpl_api/
│
├── app.py                   # Main application file
├── simulation_scenarios.py  # Predefined simulation scenarios
├── requirements.txt         # Python dependencies
├── swf_fusion_store.sql     # SQL for fusion store creation
├── swf_staging_store.sql    # SQL for staging store creation
├── swf_alliance_store.sql   # SQL for alliance store creation
├── tests/
│   ├── unit/                # Unit tests directory
│   └── integration/         # Integration tests directory
└── README.md                # Project README file
       

# Contributors
Aleksejs Vesjolijs
PhD Candidate, 
Transport and Telecommunication Institute, Riga