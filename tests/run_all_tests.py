#!/usr/bin/env python3

"""
Test runner for all automated tests
Runs unit tests and optionally smoke tests
"""

import sys
import os
import importlib.util

def run_test_file(test_file_path):
    """Run a single test file and return True if it passes"""
    try:
        # Import and run the test
        spec = importlib.util.spec_from_file_location("test_module", test_file_path)
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def run_all_tests(include_smoke=False):
    """Run all unit tests and optionally smoke tests"""
    
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Unit tests (fast)
    unit_tests = [
        'test_index_metadata_presence.py',
        'test_llm_response_shape.py', 
        'test_budget_enforcement.py'
    ]
    
    print("🧪 Running Unit Tests")
    print("=" * 40)
    
    unit_results = []
    for test_file in unit_tests:
        test_path = os.path.join(test_dir, test_file)
        if os.path.exists(test_path):
            print(f"\n📋 Running {test_file}...")
            success = run_test_file(test_path)
            unit_results.append((test_file, success))
            if success:
                print(f"✅ {test_file} passed")
            else:
                print(f"❌ {test_file} failed")
        else:
            print(f"⚠️  {test_file} not found")
            unit_results.append((test_file, False))
    
    # Smoke test (requires Ollama)
    if include_smoke:
        print(f"\n🔥 Running Smoke Tests")
        print("=" * 40)
        smoke_test_path = os.path.join(test_dir, 'smoke_test.py')
        if os.path.exists(smoke_test_path):
            print(f"\n📋 Running smoke_test.py...")
            smoke_success = run_test_file(smoke_test_path)
            if smoke_success:
                print(f"✅ smoke_test.py passed")
            else:
                print(f"❌ smoke_test.py failed")
        else:
            print("⚠️  smoke_test.py not found")
            smoke_success = False
    else:
        print(f"\n🔥 Skipping Smoke Tests (use --smoke to include)")
        smoke_success = None
    
    # Summary
    print(f"\n📊 Test Results Summary")
    print("=" * 40)
    
    passed = sum(1 for _, success in unit_results if success)
    total = len(unit_results)
    
    print(f"Unit Tests: {passed}/{total} passed")
    for test_file, success in unit_results:
        status = "✅" if success else "❌"
        print(f"  {status} {test_file}")
    
    if include_smoke:
        smoke_status = "✅" if smoke_success else "❌"
        print(f"Smoke Test: {smoke_status} smoke_test.py")
    
    # Overall result
    all_unit_passed = all(success for _, success in unit_results)
    overall_success = all_unit_passed and (smoke_success is None or smoke_success)
    
    if overall_success:
        print(f"\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    include_smoke = '--smoke' in sys.argv
    
    print("🚀 Motorcycle Recommendation System - Test Suite")
    print("=" * 50)
    
    exit_code = run_all_tests(include_smoke)
    sys.exit(exit_code)
