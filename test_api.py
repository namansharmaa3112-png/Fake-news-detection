import requests

# Test health
health = requests.get('http://localhost:5000/health')
print('Health:', health.json())

# Test prediction
text = '''The United Nations has called for immediate action to address climate change, warning that global temperatures are rising at an unprecedented rate. Scientists from around the world have confirmed that human activities are the primary cause of this warming trend. Governments are urged to implement stricter environmental policies and reduce carbon emissions.'''

response = requests.post('http://localhost:5000/predict', json={'text': text})
print('Status:', response.status_code)
print('Response:', response.text)