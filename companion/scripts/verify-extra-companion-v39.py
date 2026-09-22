"""Check the v39 completion records, without importing a neural framework.

Copied into the companion as verify_v39_evidence.py. Program execution and reader
comprehension are different evidence; this checker contains no human-reader data.
"""
from pathlib import Path
from hashlib import sha256
import json
import math

def precision_summary(root):
    record=json.loads((root/'reviews/publication-v39/framework-precision.json').read_text(encoding='utf-8'))
    lines=['## Fresh v39 forward precision checks','',
        'These are fresh CPU executions with copies of the author\'s saved 600 test inputs and weights. They are separate from the inherited v34 receipt above. No training or gradient equivalence is claimed.','',
        '| Configuration | Framework / backend version | Input / first product / score dtype | Maximum absolute logit error | Strict 1e-11 | Practical 1e-5 |',
        '|---|---|---|---:|---|---|']
    for row in record['configurations']:
        dtype=row['observed_operation_dtypes']
        lines.append(f"| {row['configuration']} | {row['version']} / {row['backend_version']} | {dtype['input']} / {dtype['first_matmul']} / {dtype['score']} | {row['max_absolute_score_error']:.8g} | {'pass' if row['strict_score_pass'] else 'FAIL'} | {'pass' if row['practical_score_pass'] else 'FAIL'} |")
    lines += ['',
        'Every configuration preserved all 600 threshold classifications. That does not make the scores bit-identical. The strict failures above remain failures even though the separately declared practical comparison passes.',
        '',
        '`default` retains the printed Keras input and layer defaults. `float64` sets only the layer policy, retaining the default Input specification. `full64` explicitly sets both. In this installed Keras/PyTorch configuration, the directly probed first matrix product still returned float32 with float64 inputs and weights. This records observed behavior, not a diagnosis of every internal kernel or a claim about every library release.',
        '',
        'The exact versions, intermediate probes, snippet hashes, original/final checks and initial warning record are included under `reviews/publication-v39`. `verify_v39_evidence.py` checks record consistency, not fresh framework execution. Framework packages are not installed by the portable requirements. With a separately prepared matching environment, run one read-only configuration with `python -B scripts/verify-framework-precision-v39.py --configuration torch64` (or another listed configuration). Do not run the no-argument production orchestrator inside the frozen reader bundle.',
    ]
    return '\n'.join(lines)

def main(root):
    record_path=root/'reviews/publication-v39/framework-precision.json'
    record=json.loads(record_path.read_text(encoding='utf-8'))
    archive=root/'outputs/chapter-one-draft/v3/experiment/data-and-model.npz'
    assert sha256(archive.read_bytes()).hexdigest()==record['archive_sha256']
    assert sha256((root/record['source']).read_bytes()).hexdigest()==record['source_sha256']
    assert sha256((root/'scripts/verify-framework-precision-v39.py').read_bytes()).hexdigest()==record['checker_sha256']
    assert len(record['configurations'])==12 and record['all_executed']
    for row in record['configurations']:
        assert row['status']=='executed' and row['test_rows']==600
        assert row['training_executed'] is False and row['gradients_compared'] is False
        error=row['max_absolute_score_error'];assert math.isfinite(error)
        assert row['strict_score_pass']==(error<=row['strict_absolute_tolerance'])
        assert row['practical_score_pass']==(error<=row['practical_absolute_tolerance'])
        assert row['classification_disagreements']==0
        assert all(k in row['observed_operation_dtypes'] for k in ('input','first_matmul','preactivation','hidden','score'))
    strict=[r['configuration'] for r in record['configurations'] if not r['strict_score_pass']]
    assert strict==record['strict_failures'] and record['all_practical_score_checks_pass']
    assert precision_summary(root) in (root/'README.md').read_text(encoding='utf-8')
    for name in ('memory-reproduction.json','exploratory-rule-profiles.json'):
        item=json.loads((root/'reviews/publication-v39'/name).read_text(encoding='utf-8'))
        assert item['passed']
        source=root/item['execution']['command'][0]
        assert sha256(source.read_bytes()).hexdigest()==item['execution']['source_sha256']
    return dict(passed=True,framework_configurations_recorded=12,frameworks_executed_by_this_check=False,
        strict_failures_preserved=strict,all_recorded_classifications_match=True,
        scope='Evidence/hash consistency only. The companion verifier separately runs the motion and exploratory programs. No human reader results.')

if __name__=='__main__':
    print(json.dumps(main(Path(__file__).resolve().parent),indent=2))
