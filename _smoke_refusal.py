import os
import sys

# Allow running from anywhere by adding the repo root (this file's directory) to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from saber_eval_refusal import RefusalEvaluator
from config import HARMFUL_PROMPTS

ev = RefusalEvaluator(prompts=HARMFUL_PROMPTS[:3], max_new_tokens=64)
res = ev.evaluate('HuggingFaceTB/SmolLM2-135M-Instruct')

print()
print('=== Refusal smoke result ===')
print('n prompts:', res.n_prompts)
print('refusal_rate_kw:', res.refusal_rate_kw)
print('refusal_rate_judge:', res.refusal_rate_judge)
for it in res.items:
    print(' ---')
    print('  prompt :', it['prompt'][:60])
    print('  refusal:', it['kw_refusal'], 'hits:', it['kw_hits'])
    print('  gen    :', it['generation'][:100].replace(chr(10), ' / '))
print('PASSED')
