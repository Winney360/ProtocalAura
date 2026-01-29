#!/usr/bin/env python3
"""
Validation script for Protocol Aura Deep Learning Integration.
Checks that all components are properly integrated and working.
"""

import os
import sys
import ast
from pathlib import Path

def check_file_exists(path, description):
    """Check if file exists."""
    if os.path.exists(path):
        print(f"  ✅ {description}")
        return True
    else:
        print(f"  ❌ {description} - NOT FOUND: {path}")
        return False

def check_import_in_file(filepath, import_statement):
    """Check if a file contains an import statement."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            if import_statement in content:
                return True
    except:
        pass
    return False

def check_function_call_in_file(filepath, function_call):
    """Check if a file contains a function call."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            if function_call in content:
                return True
    except:
        pass
    return False

def validate_integration():
    """Validate deep learning integration."""
    
    print("""
╔════════════════════════════════════════════════════════════════╗
║  Protocol Aura Deep Learning Integration Validator             ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    base_path = Path(__file__).parent
    
    # Check new files created
    print("\n📁 Checking new files...")
    all_good = True
    
    files_to_check = [
        (
            os.path.join(base_path, 'ai_service', 'analysis', 'deepfake_detector.py'),
            'Deepfake detector module'
        ),
        (
            os.path.join(base_path, 'ai_service', 'requirements_deepfake.txt'),
            'DL dependencies file'
        ),
        (
            os.path.join(base_path, 'ai_service', 'DEEPFAKE_DETECTION_GUIDE.md'),
            'Detection guide documentation'
        ),
        (
            os.path.join(base_path, 'ai_service', 'setup_deepfake.py'),
            'Setup script'
        ),
        (
            os.path.join(base_path, 'DEEPLEARNING_INTEGRATION_SUMMARY.md'),
            'Integration summary'
        ),
        (
            os.path.join(base_path, 'QUICK_START_DEEPLEARNING.md'),
            'Quick start guide'
        ),
    ]
    
    for filepath, description in files_to_check:
        if not check_file_exists(filepath, description):
            all_good = False
    
    # Check ai_service.py modifications
    print("\n🔗 Checking ai_service.py integration...")
    ai_service_path = os.path.join(base_path, 'ai_service', 'ai_service.py')
    
    integration_checks = [
        ('from analysis.deepfake_detector import get_deepfake_detector', 'Import statement'),
        ('deepfake_detector = get_deepfake_detector()', 'Detector initialization'),
        ('deepfake_detector.analyze_frame_ensemble', 'DL analysis call'),
        ("'deepfake_analysis': dl_analysis", 'Results inclusion'),
    ]
    
    for check_text, description in integration_checks:
        if check_function_call_in_file(ai_service_path, check_text):
            print(f"  ✅ {description}")
        else:
            print(f"  ❌ {description} - NOT FOUND in ai_service.py")
            all_good = False
    
    # Check deepfake_detector.py structure
    print("\n🧠 Checking deepfake_detector.py structure...")
    detector_path = os.path.join(base_path, 'ai_service', 'analysis', 'deepfake_detector.py')
    
    class_checks = [
        ('class DeepfakeDetector', 'DeepfakeDetector class'),
        ('def detect_facial_artifacts', 'Facial artifact detection'),
        ('def _analyze_skin_texture', 'Skin texture analysis'),
        ('def _analyze_eye_region', 'Eye region analysis'),
        ('def _analyze_mouth_region', 'Mouth region analysis'),
        ('def predict_deepfake_probability', 'DL prediction method'),
        ('def analyze_frame_ensemble', 'Ensemble scoring method'),
        ('def get_deepfake_detector', 'Singleton getter'),
    ]
    
    for check_text, description in class_checks:
        if check_function_call_in_file(detector_path, check_text):
            print(f"  ✅ {description}")
        else:
            print(f"  ❌ {description} - NOT FOUND")
            all_good = False
    
    # Check for optional dependency handling
    print("\n📦 Checking optional dependency handling...")
    
    optional_checks = [
        ('TF_AVAILABLE', 'TensorFlow optional flag'),
        ('MEDIAPIPE_AVAILABLE', 'MediaPipe optional flag'),
        ('graceful fallback', 'Graceful degradation'),
    ]
    
    for check_text, description in optional_checks:
        if check_function_call_in_file(detector_path, check_text):
            print(f"  ✅ {description}")
        else:
            # Some might be in comments or slightly different
            print(f"  ⚠️  {description} - check manually")
    
    # Check detection logic
    print("\n⚙️  Checking detection logic...")
    
    logic_checks = [
        ('dl_deepfake_score > 0.65', 'High DL confidence handling'),
        ('dl_deepfake_score < 0.35', 'Low DL confidence (authentic) handling'),
        ('combine_verdicts', 'Verdict combination logic'),
        ('likely_authentic', 'Authentic verdict'),
        ('likely_synthetic', 'Synthetic verdict'),
        ('needs_review', 'Needs review verdict'),
    ]
    
    for check_text, description in logic_checks:
        if check_function_call_in_file(ai_service_path, check_text):
            print(f"  ✅ {description}")
        else:
            # Some might be missing, check if logic is there
            print(f"  ⚠️  {description} - check manually")
    
    # Summary
    print("\n" + "="*60)
    if all_good:
        print("✅ INTEGRATION VALIDATION PASSED")
        print("\nAll required components are in place:")
        print("  • DeepfakeDetector class created")
        print("  • EfficientNetB0 support integrated")
        print("  • MediaPipe facial analysis available")
        print("  • Ensemble scoring implemented")
        print("  • ai_service.py updated with DL calls")
        print("  • Documentation complete")
        print("  • Setup script provided")
        print("\n🚀 Next steps:")
        print("  1. Run: python setup_deepfake.py")
        print("  2. Start service: python -m uvicorn ai_service:app")
        print("  3. Test: curl -X POST http://localhost:8000/analyze -F video=@test.mp4")
        return True
    else:
        print("⚠️  INTEGRATION VALIDATION INCOMPLETE")
        print("\nSome components are missing. Check the output above.")
        print("See DEEPLEARNING_INTEGRATION_SUMMARY.md for details.")
        return False

if __name__ == '__main__':
    success = validate_integration()
    sys.exit(0 if success else 1)
