# Water Fraud Detection System (Master's Thesis)

This repository contains the core analytical scripts developed for my Master's Thesis (2019) at **FEUP**, in collaboration with **AdP (Águas de Portugal)**. The project focuses on identifying illegal water consumption and contractual fraud.

## Project Overview
The tool was designed to optimize field inspections by predicting which customers have a high probability of illegal connections (bypassed meters or fraud), significantly reducing operational costs and non-revenue water (NRW) losses.


## The Hybrid Approach

### Phase 1: Heuristic Modeling (Rules-Based)
* Uses a **6th-degree polynomial function** to calculate the probability of re-offending based on historical illicit records.
* Integrates business rules for suspended contracts with active flow detection.

### Phase 2: Machine Learning Evolution
* Implementation of a **Decision Tree (DT)** model to automate classification.
* **Features used:** Contract state, consumption trends (2018-2019), meter physical status, and housing occupancy.

## Requirements
* **Python 3.x**
* **Pandas & NumPy** (Data processing)
* **Scikit-learn & Joblib** (ML model deployment)

## Academic Results
The methodology was tested against real datasets, providing a prioritized list of technical inspections. It demonstrated that a hybrid model (Rules + ML) is more resilient to data noise in public utility sectors.
