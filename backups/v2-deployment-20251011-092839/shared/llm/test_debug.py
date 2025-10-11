import ollama

client = ollama.Client(host='http://localhost:11434')
response = client.list()

print("Response type:", type(response))
print("\nResponse content:")
print(response)

if isinstance(response, dict) and 'models' in response:
    print("\nModels found:")
    for m in response['models']:
        print(f"  - {m.get('name', 'NO NAME')} | {m.get('model', 'NO MODEL')}")
