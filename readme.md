SARS-CoV-2 Mpro Hybrid AI-Quantum Drug Discovery Pipeline

Backend-First Project Note: This project is heavily backend-driven, focusing on heavy computational data processing, classical docking, machine learning, and quantum circuit simulation. The frontend interfaces (Streamlit & Interactive Flow Tree) were developed purely to visualize our backend execution outputs and display real data step-by-step.

🚀 1. Project Overview & Architecture

Traditional drug discovery is bogged down by slow screening processes and high failure rates. Classical docking algorithms are fast, but they suffer from severe electronic blind spots—failing to properly model electronic polarization, charge transfer, and covalent bonding.

Our project solves this by introducing a Hybrid Quantum-Classical Architecture:

Backend Processing: Heavy data pipelines ingest molecular libraries, handle feature extraction using RDKit, execute classical docking, run Random Forest machine learning models, and compute electronic corrections via quantum simulation.

Quantum Rescoring Layer: We leverage Qiskit Aer to compute true physical and electronic parameters at the molecular binding interface.

🛠️ 2. Frontend Purpose

While the core muscle of this project resides in its backend scripts and data pipelines, we built clean, interactive frontend visualization tools:

Interactive Flow Tree (flow_tree.html): A step-by-step animated visual guide to trace the data journey during presentations.

Streamlit Dashboard (app.py): Connects directly to local project data sheets (mpro_testset.csv, pilot_scores.csv, and final_quantum_shortlist.csv) to let judges and users inspect real dataset tables, metrics, and molecule IDs dynamically.

🔄 3. Pipeline Flow Summary

Molecules move through a strict, multi-tiered funnel architecture:

Phase 01 - 3D Structure Initialization: Loading spatial coordinate geometry of the SARS-CoV-2 Main Protease (Mpro) target protein bound with the reference drug (Nirmatrelvir variant).

Phase 02 - Master Dataset Composition: Aggregating 1,100 total molecules consisting of 100 real active Mpro inhibitors and 1,000 engineered decoys, validated for property fairness (AUC $\approx$ 0.50).

Phase 03 - Auto-Docking Filter: High-throughput classical docking filters down the pool. (Classical-only scoring places our reference drug down at Rank #44 due to electronic blind spots).

Phase 04 - Random Forest ML Screening: Machine learning fingerprint analysis rapidly filters chemical features and predicts initial binding affinities.

Phase 05 - Qiskit Aer Quantum Rescoring: Quantum circuit simulation models true electronic polarization and electrostatic stabilization.

Phase 06 - Final Elite Top 30 Shortlist: Generates the final high-priority candidates.

🏆 4. Validation & Key Results

Proof of Concept: Using COVID-19 Main Protease as our benchmark, we tested our system with zero hardcoded advantages or biased training inputs.

The Outcome: After passing 1,100 molecules through our hybrid pipeline, we successfully filtered them down to an elite Top 30 shortlist. Crucially, this shortlist includes our known COVID-19 reference drug—proving the system works. These candidates are now primed and ready to be handed over to lab scientists for physical wet-lab validation.

📂 5. Project Structure & Setup Instructions

Directory Structure

├── app.py                     # Streamlit frontend dashboard
├── flow_tree.html             # Interactive scrollytelling workflow tree
├── docking/
│   ├── pilot_scores.csv       # Auto-docking pilot record outputs
│   └── final_quantum_shortlist.csv # Final elite Top 30 quantum results
└── testset/
    └── mpro_testset.csv       # Master active and decoy dataset (1,100 molecules)