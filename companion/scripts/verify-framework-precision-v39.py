"""Fresh forward-only checks in isolated processes, with observed intermediate dtypes.

Native snippets are executed verbatim with copied saved weights. Keras alternatives
only substitute the documented backend; the controlled float64 variants additionally
set a dtype policy before building the same layers. No training or gradient test.
"""
from pathlib import Path
from hashlib import sha256
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
import re
import subprocess
import sys

B = Path(__file__).resolve().parents[1]
R = B/'reviews/publication-v39'
SOURCE = B/'manuscript/chapters/v39-01-neural-networks/notes-v39.tex'
ARCHIVE = B/'outputs/chapter-one-draft/v3/experiment/data-and-model.npz'
CONFIGS = ['torch64','tensorflow64','jax64'] + [f'keras-{b}-{d}' for b in ('tensorflow','jax','torch') for d in ('default','float64','full64')]

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def execute(config):
    import numpy as np
    source = SOURCE.read_text(encoding='utf-8')
    kind = config.split('-')[0].removesuffix('64')
    snippets = re.findall(r'\\begin\{lstlisting\}\[language=Python\]\n(.*?)\\end\{lstlisting\}',source,re.S)
    snippet = next(s for s in snippets if f'import {kind}' in s)
    original = snippet
    with np.load(ARCHIVE,allow_pickle=False) as archive:
        x=archive['test_x'].copy()
        W,b,v,c=[archive[k].copy() for k in ('W','b','v','c')]
    arrays=[W,b,v.reshape(-1,1),c.reshape(1)]
    expected=(np.maximum(x@W+b,0)@v+c).reshape(-1,1)
    namespace={}
    if kind=='jax' or config.startswith('keras-jax'):
        import jax
        jax.config.update('jax_enable_x64',True)
        jax.config.update('jax_platform_name','cpu')
    policy=None
    input_spec_dtypes=None
    if kind=='keras':
        backend=config.split('-')[1]
        snippet=snippet.replace('"KERAS_BACKEND"] = "tensorflow"',f'"KERAS_BACKEND"] = "{backend}"')
        if config.endswith(('float64','full64')):
            snippet=snippet.replace('import keras','import keras\nkeras.config.set_dtype_policy("float64")')
        if config.endswith('full64'):
            snippet=snippet.replace('keras.Input(shape=(9,))','keras.Input(shape=(9,), dtype="float64")')
    exec(compile(snippet,str(SOURCE),'exec'),namespace)
    if kind=='torch':
        torch=namespace['torch'];torch.set_num_threads(1)
        model=namespace['model'].double()
        with torch.no_grad():
            model[0].weight.copy_(torch.from_numpy(W.T.copy()))
            model[0].bias.copy_(torch.from_numpy(b))
            model[2].weight.copy_(torch.from_numpy(v.reshape(1,-1)))
            model[2].bias.copy_(torch.from_numpy(c.reshape(1)))
            tx=torch.from_numpy(x)
            product=tx@model[0].weight.T
            pre=model[0](tx);hidden=model[1](pre);score=model(tx)
            actual=score.numpy()
        weights=[str(p.dtype) for p in model.parameters()]
        dtypes={'input':str(tx.dtype),'first_matmul':str(product.dtype),'preactivation':str(pre.dtype),'hidden':str(hidden.dtype),'score':str(score.dtype)}
        version=str(torch.__version__);backend='torch';backend_version=version
    elif kind=='tensorflow':
        tf=namespace['tf']
        params=[tf.convert_to_tensor(a) for a in arrays];tx=tf.convert_to_tensor(x)
        product=tf.linalg.matmul(tx,params[0]);pre=product+params[1];hidden=tf.nn.relu(pre)
        score=namespace['tensorflow_score'](tx,params);actual=score.numpy()
        weights=[str(p.dtype.name) for p in params]
        dtypes={k:str(t.dtype.name) for k,t in [('input',tx),('first_matmul',product),('preactivation',pre),('hidden',hidden),('score',score)]}
        version=str(tf.__version__);backend='tensorflow';backend_version=version
    elif kind=='jax':
        jax,jnp=namespace['jax'],namespace['jnp']
        params=[jnp.asarray(a) for a in arrays];tx=jnp.asarray(x)
        product=jnp.matmul(tx,params[0]);pre=product+params[1];hidden=jax.nn.relu(pre)
        score=namespace['jax_score'](params,tx);actual=np.asarray(score)
        weights=[str(p.dtype) for p in params]
        dtypes={k:str(t.dtype) for k,t in [('input',tx),('first_matmul',product),('preactivation',pre),('hidden',hidden),('score',score)]}
        version=str(jax.__version__);backend='jax';backend_version=version
    else:
        keras=namespace['keras'];model=namespace['model'];ops=keras.ops
        model.set_weights(arrays)
        tx=ops.convert_to_tensor(x,dtype=model.compute_dtype)
        product=ops.matmul(tx,model.layers[0].kernel)
        pre=ops.add(product,model.layers[0].bias)
        hidden=model.layers[0](tx)
        score=model(tx,training=False)
        actual=ops.convert_to_numpy(score)
        weights=[str(w.dtype) for w in model.weights]
        dtypes={k:str(t.dtype) for k,t in [('input',tx),('first_matmul',product),('preactivation',pre),('hidden',hidden),('score',score)]}
        version=str(keras.__version__);backend=keras.backend.backend()
        backend_module=__import__(backend)
        backend_version=str(backend_module.__version__)
        policy=str(model.dtype_policy)
        input_spec_dtypes=[str(t.dtype) for t in model.inputs]
    error=float(np.max(np.abs(actual-expected)))
    expected_probability=np.exp(-np.logaddexp(0,-expected))
    actual_probability=np.exp(-np.logaddexp(0,-actual.astype(np.float64)))
    disagreement=int(np.count_nonzero((actual>=0)!=(expected>=0)))
    assert actual.shape==expected.shape==(600,1) and np.isfinite(actual).all()
    return dict(configuration=config,status='executed',framework=kind,version=version,
        backend=backend,backend_version=backend_version,device='CPU',numpy_version=np.__version__,
        test_rows=600,input_archive_dtype=str(x.dtype),copied_archive_weight_dtypes=[str(a.dtype) for a in arrays],
        actual_weight_dtypes=weights,observed_operation_dtypes=dtypes,requested_policy=policy,
        keras_input_spec_dtypes=input_spec_dtypes,
        keras_float64_variant='full64 also sets the Input specification explicitly; float64 sets only the layer policy, preserving the printed Input default' if kind=='keras' else None,
        intermediate_scope='Direct first matrix-product/preactivation probes and actual layer outputs. This is not a trace of every backend kernel.',
        score_shape=list(actual.shape),score_dtype=str(actual.dtype),max_absolute_score_error=error,
        reference='NumPy float64 forward logits from author-archived arrays, not reviewer-reconstructed weights',
        strict_absolute_tolerance=1e-11,strict_score_pass=error<=1e-11,
        practical_absolute_tolerance=1e-5,practical_score_pass=error<=1e-5,
        tolerance_policy='Both thresholds declared before running. Strict failure is retained even when the practical comparison passes.',
        sigmoid_probability_error=float(np.max(np.abs(actual_probability-expected_probability))),
        sigmoid_scope='NumPy float64 postprocessing of both score arrays, not a backend-specific sigmoid check',
        classification_disagreements=disagreement,threshold='score >= 0',
        printed_snippet_sha256=sha256(original.encode()).hexdigest(),executed_snippet_sha256=sha256(snippet.encode()).hexdigest(),
        exact_printed_snippet=snippet==original,
        setup='Copied frozen parameters, matching transpose/column conventions. Native models use float64. Keras default and explicit float64 are recorded separately.',
        training_executed=False,gradients_compared=False)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--configuration',choices=CONFIGS)
    args=parser.parse_args()
    if args.configuration:
        print(json.dumps(execute(args.configuration),indent=2));return
    before=digest(ARCHIVE)
    directory=R/'framework-precision';directory.mkdir(exist_ok=True)
    environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',JAX_PLATFORMS='cpu',JAX_ENABLE_X64='true',CUDA_VISIBLE_DEVICES='-1',TF_CPP_MIN_LOG_LEVEL='2')
    def run(config):
        result=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--configuration',config],
            text=True,capture_output=True,env=environment,timeout=120)
        (directory/(config+'.stderr.txt')).write_text(result.stderr,encoding='utf-8')
        (directory/(config+'.stdout.txt')).write_text(result.stdout,encoding='utf-8')
        record=json.loads(result.stdout) if result.returncode==0 else dict(configuration=config,status='execution_failed',returncode=result.returncode,stderr=result.stderr)
        (directory/(config+'.json')).write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        print(config+': '+record['status'],flush=True)
        return record
    with ThreadPoolExecutor(max_workers=2) as pool:
        records=list(pool.map(run,CONFIGS))
    assert digest(ARCHIVE)==before
    installed=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (R/'framework-environment-lock.txt').write_text(installed,encoding='utf-8')
    report=dict(edition='v39',archive_sha256=before,archive_unchanged=True,source=SOURCE.relative_to(B).as_posix(),
        source_sha256=digest(SOURCE),checker_sha256=digest(Path(__file__)),executable=sys.executable,python=sys.version,
        scope='Fresh author-archive CPU forward checks only, no training or reviewer-array substitution.',
        configurations=records,all_executed=all(r['status']=='executed' for r in records),
        all_practical_score_checks_pass=all(r.get('practical_score_pass',False) for r in records),
        all_classifications_match=all(r.get('classification_disagreements')==0 for r in records),
        strict_failures=[r['configuration'] for r in records if r.get('strict_score_pass') is False])
    (R/'framework-precision.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='configurations'},indent=2))

if __name__=='__main__':
    main()
