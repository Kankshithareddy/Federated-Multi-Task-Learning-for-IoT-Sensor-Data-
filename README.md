# Federated-Multi-Task-Learning-for-IoT-Sensor-Data

A privacy-preserving deep learning framework that performs simultaneous classification and regression on distributed IoT device data using Federated Averaging (FedAvg).

# Overview

This project implements a multi-task federated learning setup where IoT clients train locally on their data and share only model weights, not raw samples. A shared encoder learns global device patterns, while two task-specific heads perform:

Device Family Classification
Magnitude Regression

# Key Features

Federated architecture using central aggregation and decentralized client training
Unified shared encoder for representation learning
Task-specific loss balancing for stable joint optimization
Scaled vs. real-value regression handling
Automatic reporting of metrics and sample predictions

# Model Performance

Classification
High accuracy across all IoT families (consistent convergence across rounds)

Regression (Real Scale)
Scaled MSE: 0.00051
MAE: 0.2418
Predictions typically within ≤ 0.2–0.3 of true magnitude values

# Project Structure
MultiTask_Reg.ipynb # Complete notebook: preprocessing, model, FL loop, evaluation

# Tech Stack

Python, PyTorch
Federated Averaging (FedAvg)
NumPy, Pandas, Scikit-Learn
Matplotlib for reporting

# How It Works

Data is split into multiple IoT "clients"
Each client trains the shared encoder + two heads locally
Server aggregates weights using FedAvg
