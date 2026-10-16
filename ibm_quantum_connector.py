import os
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

def run_quantum_circuit_execution(circuit: QuantumCircuit, use_real_hardware: bool = False, api_token: str = None):
    """
    Executes a quantum circuit either on a local simulator or dispatches 
    it to a real IBM Quantum Hardware processor in the cloud.
    """
    if not use_real_hardware:
        # Fallback to local statevector / Aer simulator (Instant execution for demo)
        from qiskit.primitives import StatevectorSampler
        print("[INFO] Running on Local Qiskit Simulator...")
        sampler = StatevectorSampler()
        job = sampler.run([circuit])
        result = job.result()
        return {"mode": "Simulator", "data": str(result)}

    else:
        # Real IBM Quantum Hardware Execution
        try:
            print("[INFO] Connecting to IBM Quantum Platform...")
            
            # Save account token if provided dynamically
            if api_token:
                QiskitRuntimeService.save_account(token=api_token, overwrite=True)
            
            # Initialize runtime service
            service = QiskitRuntimeService()
            
            # Get the least busy real quantum backend available on your free tier
            backend = service.least_busy(simulator=False, operational=True)
            print(f"[SUCCESS] Dispatched job to real IBM Hardware: {backend.name}")
            
            # Transpile circuit for the specific hardware topology
            transpiled_circuit = transpile(circuit, backend=backend)
            
            # Run using Qiskit Runtime Sampler Primitive
            sampler = Sampler(backend)
            job = sampler.run([transpiled_circuit])
            print(f"[INFO] Job ID on IBM QPU: {job.job_id()}")
            
            result = job.result()
            return {"mode": f"Real IBM QPU ({backend.name})", "job_id": job.job_id(), "data": str(result)}
            
        except Exception as e:
            print(f"[ERROR] Failed to connect to IBM QPU: {e}")
            return {"mode": "Error", "data": str(e)}