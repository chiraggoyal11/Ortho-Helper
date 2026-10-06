#!/usr/bin/env python3
"""
Real-time Training Monitor for Parallel Medical AI Training
Tracks both cardiomegaly and knee arthritis model training progress
"""

import os
import time
import psutil
from datetime import datetime

def get_python_processes():
    """Get all Python processes with their details"""
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent', 'create_time']):
        try:
            if proc.info['name'] == 'python.exe':
                python_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return python_processes

def check_model_files():
    """Check for newly created model files"""
    models_dir = "models"
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5') and 'fast' in f and '95' in f]
        return model_files
    return []

def format_memory(bytes_val):
    """Format memory in human readable format"""
    mb = bytes_val / (1024 * 1024)
    if mb > 1024:
        return f"{mb/1024:.1f} GB"
    return f"{mb:.0f} MB"

def monitor_training():
    """Monitor training progress"""
    print("🔍 PARALLEL MEDICAL AI TRAINING MONITOR")
    print("=" * 60)
    print("📊 Tracking: Cardiomegaly + Knee Arthritis Training")
    print("🎯 Target: 95%+ accuracy for both models")
    print("=" * 60)
    
    start_time = time.time()
    
    while True:
        try:
            current_time = datetime.now().strftime("%H:%M:%S")
            elapsed = time.time() - start_time
            
            print(f"\n⏰ {current_time} | Runtime: {elapsed/60:.1f} minutes")
            print("-" * 50)
            
            # Check Python processes
            python_procs = get_python_processes()
            
            if len(python_procs) >= 2:
                print(f"✅ {len(python_procs)} Python training processes detected")
                
                for i, proc in enumerate(python_procs):
                    memory = format_memory(proc.info['memory_info'].rss)
                    cpu = proc.cpu_percent(interval=0.1)
                    runtime = (time.time() - proc.info['create_time']) / 60
                    
                    # Identify likely training process by memory usage
                    if proc.info['memory_info'].rss > 500 * 1024 * 1024:  # > 500MB
                        model_type = "🫀 Cardiomegaly" if proc.info['memory_info'].rss > 1000 * 1024 * 1024 else "🦴 Knee Arthritis"
                    else:
                        model_type = f"⚡ Process {i+1}"
                    
                    print(f"  {model_type} (PID: {proc.pid})")
                    print(f"    💾 Memory: {memory} | 🔥 CPU: {cpu:.1f}% | ⏱️  Runtime: {runtime:.1f}m")
                
            else:
                print(f"⚠️  Only {len(python_procs)} Python process(es) detected")
                if len(python_procs) == 1:
                    print("   Single training may be running - checking for completion...")
                elif len(python_procs) == 0:
                    print("   No Python processes found - training may have completed!")
            
            # Check for completed models
            model_files = check_model_files()
            if model_files:
                print(f"\n🎉 COMPLETED MODELS DETECTED:")
                for model in model_files:
                    file_size = os.path.getsize(f"models/{model}") / (1024 * 1024)
                    print(f"  ✅ {model} ({file_size:.1f} MB)")
            
            # Check if training completed
            if len(python_procs) == 0 and model_files:
                print(f"\n🏁 TRAINING COMPLETED! Found {len(model_files)} trained models")
                break
            
            time.sleep(30)  # Update every 30 seconds
            
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"⚠️  Monitor error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    try:
        monitor_training()
    except Exception as e:
        print(f"❌ Monitor failed: {e}")