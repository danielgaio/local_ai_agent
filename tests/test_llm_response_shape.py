#!/usr/bin/env python3

"""
Test LLM response shape handling
Verifies that analyze_with_llm correctly handles valid/invalid JSON responses
"""

import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_llm_response_shape():
    """Test analyze_with_llm behavior with valid/invalid JSON responses"""
    
    # We need to mock the imports and dependencies
    import types
    
    # Mock langchain_ollama.llms module
    ll_ms = types.SimpleNamespace()
    class _FakeOllama:
        def __init__(self, model=None):
            self.model = model
        def generate(self, msgs):
            # return a simple object similar to langchain's output
            class G:
                def __init__(self):
                    from types import SimpleNamespace
                    self.generations = [[SimpleNamespace(text=mock_llm_response)]]
            return G()

    ll_ms.OllamaLLM = _FakeOllama
    sys.modules['langchain_ollama'] = types.SimpleNamespace()
    sys.modules['langchain_ollama.llms'] = ll_ms
    
    # Mock vector module with a dummy retriever
    class DummyRetriever:
        def get_relevant_documents(self, q):
            return []
    sys.modules['vector'] = types.SimpleNamespace(retriever=DummyRetriever())
    
    # Import main module after mocking dependencies
    import main
    
    # Test cases with different response types
    test_cases = [
        {
            'name': 'Valid recommendation JSON',
            'response': '{"type": "recommendation", "picks": [{"brand": "BMW", "model": "R1250GS", "year": 2023, "price_est": 17995, "reason": "excellent suspension for touring", "evidence": "WP suspension"}]}',
            'should_parse': True,
            'expected_type': 'recommendation'
        },
        {
            'name': 'Valid clarifying question JSON',
            'response': '{"type": "clarify", "question": "What type of riding do you plan to do?"}',
            'should_parse': True,
            'expected_type': 'clarify'
        },
        {
            'name': 'Invalid JSON - missing quotes',
            'response': '{type: recommendation, picks: []}',
            'should_parse': False,
            'expected_type': None
        },
        {
            'name': 'Invalid JSON - trailing comma',
            'response': '{"type": "recommendation", "picks": [],}',
            'should_parse': False,
            'expected_type': None
        },
        {
            'name': 'Plain text response (no JSON)',
            'response': 'I need more information about your riding preferences.',
            'should_parse': False,
            'expected_type': None
        },
        {
            'name': 'Empty response',
            'response': '',
            'should_parse': False,
            'expected_type': None
        }
    ]
    
    print("=== Testing LLM Response Shape Handling ===")
    
    conversation_history = ["I need a bike with good suspension"]
    top_reviews = []
    
    all_passed = True
    
    for i, case in enumerate(test_cases):
        print(f"\nTest case {i+1}: {case['name']}")
        print(f"Response: {case['response'][:50]}{'...' if len(case['response']) > 50 else ''}")
        
        # Set the mock response
        global mock_llm_response
        mock_llm_response = case['response']
        
        # Mock the invoke function to return our test response
        def mock_invoke_model_with_prompt(prompt):
            return case['response']
        
        # Replace the function in main module
        original_invoke = main.invoke_model_with_prompt
        main.invoke_model_with_prompt = mock_invoke_model_with_prompt
        
        try:
            result = main.analyze_with_llm(conversation_history, top_reviews)
            
            # Check if JSON parsing worked as expected
            if case['should_parse']:
                # Should return formatted output
                is_valid_output = isinstance(result, str) and (
                    "Top recommendations:" in result or 
                    result.strip() == case['response'] or  # clarifying questions return as-is
                    "?" in result  # questions
                )
                print(f"  Valid parsing: {'✓' if is_valid_output else '✗'}")
                if not is_valid_output:
                    all_passed = False
                    print(f"    Expected formatted output, got: {result[:100]}...")
            else:
                # Should return raw response for invalid JSON
                is_raw_response = result == case['response']
                print(f"  Raw response returned: {'✓' if is_raw_response else '✗'}")
                if not is_raw_response:
                    all_passed = False
                    print(f"    Expected raw response, got: {result[:100]}...")
                    
            # Test JSON parsing directly
            try:
                parsed = json.loads(case['response'])
                can_parse_json = True
                parsed_type = parsed.get('type')
            except:
                can_parse_json = False
                parsed_type = None
                
            parse_matches_expectation = can_parse_json == case['should_parse']
            print(f"  JSON parsing expectation: {'✓' if parse_matches_expectation else '✗'}")
            
            if can_parse_json and case['expected_type']:
                type_matches = parsed_type == case['expected_type']
                print(f"  Type field correct: {'✓' if type_matches else '✗'}")
                if not type_matches:
                    all_passed = False
                    print(f"    Expected type: {case['expected_type']}, got: {parsed_type}")
            
        except Exception as e:
            print(f"  Error during analysis: {e}")
            all_passed = False
            
        finally:
            # Restore original function
            main.invoke_model_with_prompt = original_invoke
    
    if not all_passed:
        raise AssertionError("Some LLM response shape tests failed")
        
    print(f"\n✅ All LLM response shape tests passed!")


if __name__ == "__main__":
    test_llm_response_shape()
    print("🎉 test_llm_response_shape completed successfully!")
