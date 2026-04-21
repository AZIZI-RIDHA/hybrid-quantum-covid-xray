from qiskit_ibm_provider import IBMProvider
provider = IBMProvider()
backend = provider.get_backend('ibmq_manila')
print('Backend:', backend.name)
