import json

# Load and display registry info
registry = json.load(open('models/registry/model_registry.json'))

print('📊 REGISTRY STATUS:')
print(f'Total models: {len(registry["models"])}')
print(f'Registry version: {registry["version"]}')

print('\n🎯 NEW INTENSIVE MODELS (v2):')
for m in registry['models'].keys():
    if '_v2' in m:
        model_info = registry['models'][m]
        acc = model_info['accuracy']*100
        grade = model_info['training_info']['performance_level']
        print(f'  • {m}: {acc:.1f}% accuracy ({grade})')

print('\n🔄 ACTIVE MODELS:')
for k,v in registry['active_models'].items():
    print(f'  • {k}: {v}')

print('\n✅ Registry successfully updated with new intensive models!')