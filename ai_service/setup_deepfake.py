#!/usr/bin/env python3
"""
Setup script for Protocol Aura AI Service with Deep Learning support.
Installs dependencies and validates the environment.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a shell command and report status."""
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"✅ {description} - SUCCESS\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error: {e}\n")
        return False

def check_python_version():
    """Verify Python 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor} OK")
    return True

def check_venv():
    """Check if running in virtual environment."""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
        return True
    print("⚠️  Not running in virtual environment (optional but recommended)")
    return False

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║     Protocol Aura AI Service Setup                           ║
║     Deep Learning Deepfake Detection Integration             ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check venv
    check_venv()
    
    print("\n" + "="*60)
    print("🔍 Environment Check")
    print("="*60)
    
    # Check existing modules
    modules_to_check = [
        ('fastapi', 'FastAPI'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
    ]
    
    for module, name in modules_to_check:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ⚠️  {name} - Not installed")
    
    # Optional modules
    optional_modules = [
        ('tensorflow', 'TensorFlow (Deep Learning)'),
        ('mediapipe', 'MediaPipe (Facial Analysis)'),
    ]
    
    print("\n  Optional modules:")
    for module, name in optional_modules:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ⚠️  {name} - Not installed (recommended)")
    
    # Upgrade pip
    print("\n" + "="*60)
    response = input("Install/upgrade dependencies? (y/n): ").strip().lower()
    
    if response == 'y':
        run_command(
            f"{sys.executable} -m pip install --upgrade pip",
            "Upgrading pip"
        )
        
        # Install core dependencies
        run_command(
            f"{sys.executable} -m pip install -r requirements_deepfake.txt",
            "Installing deepfake detection dependencies"
        )
        
        # Optional: Install full ML stack
        print("\n" + "="*60)
        ml_choice = input("Install full ML stack? (TF + MediaPipe) - recommended (y/n): ").strip().lower()
        if ml_choice == 'y':
            run_command(
                f"{sys.executable} -m pip install tensorflow>=2.10.0",
                "Installing TensorFlow"
            )
            run_command(
                f"{sys.executable} -m pip install mediapipe>=0.8.9.1",
                "Installing MediaPipe"
            )
    
    # Verify installation
    print("\n" + "="*60)
    print("🧪 Verifying Installation")
    print("="*60)
    
    verify_code = """
import sys
errors = []

try:
    import fastapi
    print("✅ FastAPI available")
except ImportError:
    errors.append("FastAPI")

try:
    import cv2
    print("✅ OpenCV available")
except ImportError:
    errors.append("OpenCV")

try:
    import numpy
    print("✅ NumPy available")
except ImportError:
    errors.append("NumPy")

try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} available")
except ImportError:
    print("⚠️  TensorFlow not available (optional)")

try:
    import mediapipe as mp
    print(f"✅ MediaPipe available")
except ImportError:
    print("⚠️  MediaPipe not available (optional)")

if errors:
    print(f"\\n❌ Missing required: {', '.join(errors)}")
    sys.exit(1)
else:
    print("\\n✅ All required dependencies installed!")
"""
    
    result = subprocess.run([sys.executable, "-c", verify_code])
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print("🚀 SETUP COMPLETE!")
        print("="*60)
        print("""
You can now start the AI service with:

    python -m uvicorn ai_service:app --reload

Or to see the deep learning detection in action:

    curl -X POST http://localhost:8000/analyze \\
         -F "video=@your_test_video.mp4"

For more information, see: DEEPFAKE_DETECTION_GUIDE.md
        """)
    else:
        print("\n" + "="*60)
        print("⚠️  Setup Complete with Warnings")
        print("="*60)
        print("""
Some dependencies are missing. The service will work with
reduced functionality. Install missing packages manually or
re-run this script with the 'y' option.

For more details, see: DEEPFAKE_DETECTION_GUIDE.md
        """)

if __name__ == '__main__':
    main()
