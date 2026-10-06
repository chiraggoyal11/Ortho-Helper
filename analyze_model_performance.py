#!/usr/bin/env python3
"""
Model Accuracy Analysis Script
"""
import json
import os

def analyze_model_accuracies():
    """Analyze all models and show maximum accuracy for each condition"""
    
    registry_path = 'models/registry/model_registry.json'
    if not os.path.exists(registry_path):
        print("❌ Model registry not found!")
        return
    
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    models = registry.get('models', {})
    
    # Group models by condition
    conditions = {}
    
    for model_id, model_info in models.items():
        dataset_type = model_info.get('dataset_type', 'unknown')
        accuracy = model_info.get('accuracy', 0)
        
        # Get accuracy from performance metrics if available
        perf_metrics = model_info.get('performance_metrics', {})
        test_accuracy = perf_metrics.get('test_accuracy', 0)
        metric_accuracy = perf_metrics.get('accuracy', 0)
        
        # Use the highest available accuracy
        final_accuracy = max(accuracy, test_accuracy, metric_accuracy)
        
        if dataset_type not in conditions:
            conditions[dataset_type] = []
        
        conditions[dataset_type].append({
            'model_id': model_id,
            'accuracy': final_accuracy,
            'architecture': model_info.get('architecture', 'Unknown'),
            'is_active': model_info.get('is_active', False),
            'model_type': model_info.get('model_type', 'Standard')
        })
    
    print('📊 MAXIMUM ACCURACY FOR EACH MEDICAL CONDITION:')
    print('=' * 70)
    
    for condition, model_list in conditions.items():
        # Find the model with highest accuracy
        best_model = max(model_list, key=lambda x: x['accuracy'])
        
        print(f'\n🏥 {condition.upper().replace("_", " ")}:')
        print(f'   Max Accuracy: {best_model["accuracy"]:.4f} ({best_model["accuracy"]*100:.1f}%)')
        print(f'   Best Model: {best_model["model_id"]}')
        print(f'   Architecture: {best_model["architecture"]}')
        print(f'   Type: {best_model["model_type"]}')
        print(f'   Currently Active: {"✅" if best_model["is_active"] else "❌"}')
        
        # Show all models for this condition
        print(f'   All models ({len(model_list)}):')
        for model in sorted(model_list, key=lambda x: x['accuracy'], reverse=True):
            status = "✅ ACTIVE" if model['is_active'] else ""
            print(f'     - {model["accuracy"]*100:.1f}% ({model["architecture"]}/{model["model_type"]}) {status}')
    
    print('\n\n📈 OVERALL PERFORMANCE RANKING:')
    print('=' * 50)
    
    # Calculate best performance for each condition
    condition_best = []
    for condition, model_list in conditions.items():
        max_acc = max(model['accuracy'] for model in model_list)
        condition_best.append((condition, max_acc))
    
    # Sort by accuracy (highest first)
    condition_best.sort(key=lambda x: x[1], reverse=True)
    
    for i, (condition, accuracy) in enumerate(condition_best, 1):
        if accuracy >= 0.95:
            grade = '🟢 Excellent'
        elif accuracy >= 0.85:
            grade = '🔵 Very Good'
        elif accuracy >= 0.75:
            grade = '🟡 Good'
        elif accuracy >= 0.65:
            grade = '🟠 Fair'
        else:
            grade = '🔴 Needs Improvement'
            
        print(f'{i}. {condition.replace("_", " ").title()}: {accuracy*100:.1f}% {grade}')
    
    # Overall statistics
    avg_accuracy = sum(acc for _, acc in condition_best) / len(condition_best)
    print(f'\nℹ️  Average Performance: {avg_accuracy*100:.1f}%')
    print(f'📊 Total Models: {len(models)}')
    print(f'🎯 Conditions Covered: {len(conditions)}')
    
    # Performance categories
    excellent = sum(1 for _, acc in condition_best if acc >= 0.95)
    good = sum(1 for _, acc in condition_best if 0.85 <= acc < 0.95)
    fair = sum(1 for _, acc in condition_best if 0.65 <= acc < 0.85)
    poor = sum(1 for _, acc in condition_best if acc < 0.65)
    
    print(f'\n🏆 Performance Distribution:')
    print(f'   Excellent (≥95%): {excellent} conditions')
    print(f'   Very Good (85-94%): {good} conditions')
    print(f'   Fair (65-84%): {fair} conditions')
    print(f'   Poor (<65%): {poor} conditions')

if __name__ == "__main__":
    analyze_model_accuracies()