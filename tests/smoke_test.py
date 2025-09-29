#!/usr/bin/env python3

"""
Smoke test script that runs main.py with piped input
Verifies that the system produces expected output patterns
"""

import subprocess
import sys
import os
import time

def run_smoke_test():
    """Run smoke test with piped input to main.py"""
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Basic suspension request',
            'inputs': [
                'I need a bike with good suspension for touring',
                'q'  # quit
            ],
            'expected_patterns': [
                'Top recommendations:',
                'Evidence',
                'Reason:'
            ],
            'timeout': 30
        },
        {
            'name': 'Budget constraint request',
            'inputs': [
                'I want a motorcycle with long-travel suspension',
                'Budget $10,000',
                'q'
            ],
            'expected_patterns': [
                'Top recommendations:',
                'Price est:',
                'suspension'  # Should mention suspension in reason
            ],
            'timeout': 30
        },
        {
            'name': 'Vague request (should ask clarifying question)',
            'inputs': [
                'I want a bike',
                'q'
            ],
            'expected_patterns': [
                '?',  # Should contain a question
            ],
            'timeout': 20
        }
    ]
    
    print("=== Running Smoke Tests ===")
    print("Note: These tests require Ollama to be running with llama3.2:3b and mxbai-embed-large models")
    
    all_passed = True
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_script = os.path.join(script_dir, '..', 'main.py')
    
    for i, scenario in enumerate(test_scenarios):
        print(f"\nTest {i+1}: {scenario['name']}")
        print(f"Inputs: {scenario['inputs'][:-1]}")  # Don't show the 'q'
        
        try:
            # Prepare input string
            input_text = '\n'.join(scenario['inputs']) + '\n'
            
            # Run the main script with piped input
            process = subprocess.Popen(
                [sys.executable, main_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(main_script)
            )
            
            # Send input and get output with timeout
            try:
                stdout, stderr = process.communicate(input=input_text, timeout=scenario['timeout'])
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                print("  ⚠️  Test timed out (this may be normal if Ollama is slow)")
                all_passed = False
                continue
            
            # Check return code
            if process.returncode != 0:
                print(f"  ✗ Process exited with code {process.returncode}")
                if stderr:
                    print(f"  Error: {stderr[:200]}...")
                all_passed = False
                continue
            
            # Check for expected patterns
            print("  Checking output patterns:")
            found_patterns = 0
            
            for pattern in scenario['expected_patterns']:
                if pattern.lower() in stdout.lower():
                    print(f"    ✓ Found: '{pattern}'")
                    found_patterns += 1
                else:
                    print(f"    ✗ Missing: '{pattern}'")
            
            # Success if we found all expected patterns
            if found_patterns == len(scenario['expected_patterns']):
                print(f"  ✅ Smoke test passed ({found_patterns}/{len(scenario['expected_patterns'])} patterns found)")
            else:
                print(f"  ✗ Smoke test failed ({found_patterns}/{len(scenario['expected_patterns'])} patterns found)")
                all_passed = False
                
            # Show a snippet of the output for debugging
            print(f"  Output snippet: {stdout[:150].strip()}...")
                
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
            all_passed = False
    
    return all_passed


def check_prerequisites():
    """Check if Ollama and required models are available"""
    print("=== Checking Prerequisites ===")
    
    # Check if Ollama is running
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print("✗ Ollama is not running or not installed")
            return False
        
        models_output = result.stdout
        
        # Check for required models
        required_models = ['llama3.2:3b', 'mxbai-embed-large']
        missing_models = []
        
        for model in required_models:
            if model not in models_output:
                missing_models.append(model)
        
        if missing_models:
            print(f"✗ Missing required models: {missing_models}")
            print("  Run: ollama pull llama3.2:3b && ollama pull mxbai-embed-large")
            return False
        
        print("✅ All prerequisites met")
        return True
        
    except subprocess.TimeoutExpired:
        print("✗ Ollama command timed out - is Ollama running?")
        return False
    except FileNotFoundError:
        print("✗ Ollama not found - is it installed and in PATH?")
        return False
    except Exception as e:
        print(f"✗ Error checking Ollama: {e}")
        return False


if __name__ == "__main__":
    print("🔬 Motorcycle Recommendation System - Smoke Test")
    print("=" * 50)
    
    # Check prerequisites first
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Skipping smoke tests.")
        print("Make sure Ollama is running with required models:")
        print("  ollama pull llama3.2:3b")
        print("  ollama pull mxbai-embed-large")
        sys.exit(1)
    
    # Run smoke tests
    success = run_smoke_test()
    
    if success:
        print("\n🎉 All smoke tests passed!")
        print("The system is working correctly with piped input.")
    else:
        print("\n❌ Some smoke tests failed.")
        print("Check the output above for details.")
        sys.exit(1)
