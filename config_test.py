import json

with open('config.json', 'r') as f:
    config = json.load(f)

print(config['key1'])
print(type(config['key1']))