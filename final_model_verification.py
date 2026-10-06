import json

# Load registry and show all models  
with open('models/registry/model_registry.json', 'r') as f:
    registry = json.load(f)

models = registry.get('models', {})
print('📋 COMPLETE MODEL INVENTORY - ALL MODELS INTEGRATED UNIFORMLY')
print('=' * 70)

# Group by condition
conditions = {}
for model_id, model_info in models.items():
    condition = model_info.get('dataset_type', 'unknown')
    if condition not in conditions:
        conditions[condition] = []
    
    conditions[condition].append({
        'id': model_id,
        'name': model_info.get('model_name', 'Unknown'),
        'accuracy': model_info.get('performance_metrics', {}).get('accuracy', 0),
        'arch': model_info.get('architecture', 'Unknown'),
        'active': model_info.get('is_active', False),
        'version': model_info.get('version', 'Unknown')
    })

# Sort and display
for condition, model_list in sorted(conditions.items()):
    if condition == 'unknown':
        continue
        
    model_list.sort(key=lambda x: x['accuracy'], reverse=True)
    best_accuracy = model_list[0]['accuracy'] * 100 if model_list else 0
    
    print(f'\n🏥 {condition.upper()}: {len(model_list)} models (Best: {best_accuracy:.1f}%)')
    
    for model in model_list:
        status = '✅' if model['active'] else '  '
        acc = model['accuracy'] * 100
        arch = model['arch']
        model_id = model['id']
        print(f'   {status} {arch} - {acc:.1f}% - {model_id}')

print(f'\n📊 SUMMARY: {len(models)} total models across {len(conditions)-1} medical conditions')

# Show the new DenseNet121 cardiomegaly model specifically
print('\n🎯 NEW DENSENET121 CARDIOMEGALY MODEL:')
print('-' * 45)
for model_id, model_info in models.items():
    if 'densenet121' in model_id.lower() and 'cardiomegaly' in model_id.lower():
        acc = model_info.get('performance_metrics', {}).get('accuracy', 0) * 100
        active = '✅ ACTIVE' if model_info.get('is_active', False) else 'Inactive'
        print(f'✅ Model ID: {model_id}')
        print(f'✅ Accuracy: {acc:.2f}%')
        print(f'✅ Status: {active}')
        print(f'✅ Architecture: {model_info.get("architecture", "Unknown")}')
        print(f'✅ Integration: Successfully added to registry!')
        break