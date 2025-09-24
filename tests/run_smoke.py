import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Provide lightweight stubs for external modules that may not be installed in the test env
import types

# Stub langchain_ollama.llms.OllamaLLM
ll_ms = types.SimpleNamespace()
class _FakeOllama:
    def __init__(self, model=None):
        self.model = model
    def generate(self, msgs):
        # return a simple object similar to langchain's output
        class G:
            def __init__(self):
                from types import SimpleNamespace
                self.generations = [[SimpleNamespace(text='long-travel suspension offroad touring')]]
        return G()

ll_ms.OllamaLLM = _FakeOllama
sys.modules['langchain_ollama'] = types.SimpleNamespace()
sys.modules['langchain_ollama.llms'] = ll_ms

# Stub vector module with a dummy retriever
class DummyRetriever:
    def get_relevant_documents(self, q):
        return []

sys.modules['vector'] = types.SimpleNamespace(retriever=DummyRetriever())

import main as M
from main import generate_retriever_query

def fake_invoke(prompt_text):
    return 'long-travel suspension offroad touring'

M.invoke_model_with_prompt = fake_invoke

convo = [
    "I need a bike with big suspension for long off-road trips",
    "Budget under 10000"
]

q = generate_retriever_query(convo)
if not q or '\n' in q or not isinstance(q, str) or q.strip() == '':
    print('SMOKE TEST FAILED: invalid query ->', repr(q))
    sys.exit(2)

print('SMOKE TEST PASSED:', q)
sys.exit(0)
