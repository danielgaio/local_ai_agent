#!/usr/bin/env python3

"""
Test budget enforcement functionality
Verifies that the parse->validate pipeline correctly enforces budget constraints
"""

import re
from typing import Dict, List

from src.conversation.validation import validate_and_filter
from tests.test_utils import setup_test_dependencies

def test_budget_enforcement():
    """Test budget enforcement in the validation pipeline"""
    setup_test_dependencies()
    
    # Test cases for budget enforcement
    test_cases = [
        {
            'name': 'Budget $10k - all picks under budget',
            'parsed': {
                "type": "recommendation", 
                "picks": [
                    {"brand": "Honda", "model": "CB500X", "price_est": 7000, "reason": "good suspension", "evidence": "basic"},
                    {"brand": "Yamaha", "model": "MT-07", "price_est": 8500, "reason": "sporty suspension", "evidence": "firm"}
                ]
            },
            'conversation': ["I want a bike with good suspension", "Budget $10,000"],
            'expected_valid': True,
            'expected_picks_count': 2
        },
        {
            'name': 'Budget $10k - some picks over budget',
            'parsed': {
                "type": "recommendation",
                "picks": [
                    {"brand": "Honda", "model": "CB500X", "price_est": 7000, "reason": "good suspension", "evidence": "basic"},
                    {"brand": "BMW", "model": "R1250GS", "price_est": 18000, "reason": "excellent suspension", "evidence": "WP"},
                    {"brand": "Yamaha", "model": "MT-07", "price_est": 8500, "reason": "sporty suspension", "evidence": "firm"}
                ]
            },
            'conversation': ["I want a bike with good suspension", "Budget $10,000"],
            'expected_valid': True,
            'expected_picks_count': 2  # BMW should be filtered out
        },
        {
            'name': 'Budget $5k - no picks under budget',
            'parsed': {
                "type": "recommendation",
                "picks": [
                    {"brand": "Honda", "model": "CB500X", "price_est": 7000, "reason": "good suspension", "evidence": "basic"},
                    {"brand": "BMW", "model": "R1250GS", "price_est": 18000, "reason": "excellent suspension", "evidence": "WP"}
                ]
            },
            'conversation': ["I want a bike with good suspension", "Budget $5,000"],
            'expected_valid': True,
            'expected_picks_count': 0,  # All picks should be filtered out
            'expect_note': True
        },
        {
            'name': 'Budget 8k format - some picks under budget',
            'parsed': {
                "type": "recommendation",
                "picks": [
                    {"brand": "Honda", "model": "CB500X", "price_est": 7000, "reason": "good suspension", "evidence": "basic"},
                    {"brand": "Yamaha", "model": "MT-07", "price_est": 9500, "reason": "sporty suspension", "evidence": "firm"}
                ]
            },
            'conversation': ["I want a bike with good suspension", "Budget 8k"],
            'expected_valid': True,
            'expected_picks_count': 1  # Only Honda should remain
        },
        {
            'name': 'No budget mentioned - all picks remain',
            'parsed': {
                "type": "recommendation",
                "picks": [
                    {"brand": "Honda", "model": "CB500X", "price_est": 7000, "reason": "good suspension", "evidence": "basic"},
                    {"brand": "BMW", "model": "R1250GS", "price_est": 18000, "reason": "excellent suspension", "evidence": "WP"}
                ]
            },
            'conversation': ["I want a bike with good suspension"],
            'expected_valid': True,
            'expected_picks_count': 2
        },
        {
            'name': 'Clarifying question - no budget validation',
            'parsed': {
                "type": "clarify",
                "question": "What type of riding do you plan to do?"
            },
            'conversation': ["I want a bike", "Budget $5,000"],
            'expected_valid': True,
            'expected_picks_count': None  # No picks field for clarifying questions
        }
    ]
    
    print("=== Testing Budget Enforcement ===")
    
    all_passed = True
    
    for i, case in enumerate(test_cases):
        print(f"\nTest case {i+1}: {case['name']}")
        
        # Test the validation
        valid, result = validate_and_filter(case['parsed'], case['conversation'])
        
        # Check if validation result matches expectation
        valid_correct = valid == case['expected_valid']
        print(f"  Validation result: {'✓' if valid_correct else '✗'} (valid={valid})")
        if not valid_correct:
            all_passed = False
            print(f"    Expected valid={case['expected_valid']}, got valid={valid}")
        
        if valid and case['expected_picks_count'] is not None:
            actual_picks = result.get('picks', []) if isinstance(result, dict) else []
            picks_count_correct = len(actual_picks) == case['expected_picks_count']
            print(f"  Picks count: {'✓' if picks_count_correct else '✗'} ({len(actual_picks)}/{case['expected_picks_count']})")
            
            if not picks_count_correct:
                all_passed = False
                print(f"    Expected {case['expected_picks_count']} picks, got {len(actual_picks)}")
                
            # Check individual pick prices if budget was applied
            if 'Budget' in ' '.join(case['conversation']):
                # Extract budget from conversation
                convo_text = ' '.join(case['conversation'])
                budget = None
                m = re.search(r"\$\s*([0-9,]+(?:\.\d+)?)", convo_text)
                if not m:
                    m = re.search(r"([0-9,]+(?:\.\d+)?)[\s]*k\b", convo_text, re.IGNORECASE)
                    if m:
                        try:
                            budget = float(m.group(1).replace(",", "")) * 1000
                        except:
                            budget = None
                else:
                    try:
                        budget = float(m.group(1).replace(",", ""))
                    except:
                        budget = None
                
                if budget:
                    all_under_budget = all(pick.get('price_est', 0) <= budget for pick in actual_picks)
                    print(f"  All picks under budget ${int(budget)}: {'✓' if all_under_budget else '✗'}")
                    if not all_under_budget:
                        all_passed = False
                        for pick in actual_picks:
                            if pick.get('price_est', 0) > budget:
                                print(f"    {pick.get('brand', '')} {pick.get('model', '')} at ${pick.get('price_est', 0)} exceeds budget")
        
        # Check for explanatory note when no picks remain  
        if case.get('expect_note') and valid:
            has_note = isinstance(result, dict) and result.get('note')
            print(f"  Explanatory note present: {'✓' if has_note else '✗'}")
            if not has_note:
                all_passed = False
                print(f"    Expected explanatory note when no picks under budget")
        
        # Print actual picks for debugging
        if valid and isinstance(result, dict) and result.get('picks'):
            picks_summary = []
            for p in result.get('picks', []):
                brand = p.get('brand', '')
                model = p.get('model', '')
                price = p.get('price_est', 0)
                picks_summary.append(f"{brand} {model} (${price})")
            print(f"  Remaining picks: {picks_summary}")
    
    if not all_passed:
        raise AssertionError("Some budget enforcement tests failed")
        
    print(f"\n✅ All budget enforcement tests passed!")


if __name__ == "__main__":
    test_budget_enforcement()
    print("🎉 test_budget_enforcement completed successfully!")
