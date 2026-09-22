"""Read frozen Chapter 3 arrays; optionally reproduce both fixed training runs in RAM.

No files are written. --retrain performs new reproduction, not a new historical
experiment. The original output archives and figure files are never overwritten.
"""
from pathlib import Path
from hashlib import sha256
import argparse
import ast
import json
import os

for setting in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[setting]='1'
import numpy as np

B=Path(__file__).resolve().parents[1]
ARCHIVE=B/'outputs/chapters-three-six/v7/experiments-memory/data.npz'
RESULTS=B/'outputs/chapters-three-six/v7/experiments-memory/results.json'
PRODUCER=B/'examples/ch03_04/experiment_v7.py'

def initial(d):
    rng=np.random.default_rng(1701)
    return dict(w1=rng.normal(0,np.sqrt(2/d),(d,8)),b1=np.full(8,.3),
                w2=rng.normal(0,np.sqrt(2/8),(8,1)),b2=np.zeros(1))

def predict(x,p):
    return np.maximum(x@p['w1']+p['b1'],0)@p['w2']+p['b2']

def rmse(a,b):
    return float(np.sqrt(np.mean((np.ravel(a)-np.ravel(b))**2)))

def recipe_check():
    # Load only literal CONFIG and the initialization function AST. No module
    # import, plotting library or output-writing producer function is executed.
    source=PRODUCER.read_text(encoding='utf-8');tree=ast.parse(source)
    nodes=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONFIG' for t in node.targets):
            nodes.append(node)
        elif isinstance(node,ast.FunctionDef) and node.name=='initialize':
            nodes.append(node)
    assert len(nodes)==2
    namespace={'np':np}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(PRODUCER),'exec'),namespace)
    for d in (1,2):
        supplied=namespace['initialize'](d)
        assert all(np.array_equal(a,supplied[k]) for k,a in initial(d).items())
    return namespace['CONFIG']

def reproduce(x,y,d):
    params=initial(d);curve=[]
    for step in range(10001):
        a=x@params['w1']+params['b1'];h=np.maximum(a,0)
        error=h@params['w2']+params['b2']-y
        if step%25==0:
            curve.append((step,float(np.sqrt(np.mean(error**2)))))
        if step==10000:
            break
        dz=error/len(x);da=(dz@params['w2'].T)*(a>0)
        gradients={'w1':x.T@da,'b1':da.sum(axis=0),'w2':h.T@dz,'b2':dz.sum(axis=0)}
        for k in params:
            params[k]-=.03*gradients[k]
    return params,np.asarray(curve)

def main(retrain=False):
    before={str(p.relative_to(B)):sha256(p.read_bytes()).hexdigest() for p in (ARCHIVE,RESULTS,PRODUCER)}
    configuration=recipe_check();historical=json.loads(RESULTS.read_text(encoding='utf-8'))
    assert configuration==historical['configuration']
    with np.load(ARCHIVE,allow_pickle=False) as archive:
        data={k:archive[k].copy() for k in archive.files}
    for split,seed,n in [('train',7321,256),('test',9248,512)]:
        q=np.repeat(np.random.default_rng(seed).uniform(-2,2,n),2)
        velocity=np.tile(np.array([-1.,1.]),n)
        for key,value in [('current',q),('previous',q-velocity),('velocity',velocity),('following',q+velocity)]:
            assert np.array_equal(value,data[split+'_'+key]),(split,key)
    models={}
    for name,d,prefix in [('current_only',1,'current'),('history',2,'history')]:
        params={k:data[prefix+'_model_'+k] for k in ('w1','b1','w2','b2')}
        row={}
        for split in ('train','test'):
            x=(data[split+'_current'][:,None] if d==1 else np.column_stack((data[split+'_previous'],data[split+'_current'])))/3
            y=data[split+'_following'][:,None]
            output=predict(x,params)
            row[split+'_rmse']=rmse(output,y)
            assert abs(row[split+'_rmse']-historical['models'][name][split+'_rmse'])<1e-12
            if split=='test':
                row['saved_prediction_max_error']=float(np.max(np.abs(output.ravel()-data['test_prediction_'+prefix])))
                assert row['saved_prediction_max_error']<1e-12
            if retrain and split=='train':
                trained,curve=reproduce(x,y,d)
                error=max(float(np.max(np.abs(trained[k]-params[k]))) for k in params)
                curve_error=float(np.max(np.abs(curve-data[prefix+'_learning_curve'])))
                assert error<1e-11 and curve_error<1e-11,(name,error,curve_error)
                row['new_reproduction']={'updates':10000,'final_parameter_max_error':error,'learning_curve_max_error':curve_error,'absolute_tolerance':1e-11}
        models[name]=row
    exact=2*data['test_current']-data['test_previous']
    exact_error=rmse(exact,data['test_following']);assert exact_error==0
    assert all(sha256((B/p).read_bytes()).hexdigest()==h for p,h in before.items())
    return dict(passed=True,archive_sha256=before[str(ARCHIVE.relative_to(B))],source_hashes=before,
        python_numpy_version=np.__version__,configuration=configuration,
        initialization={'generator':'numpy.random.default_rng(1701), restarted for each input dimension',
            'W1':'normal(0, sqrt(2/d)), shape (d,8), drawn first','b1':'eight entries equal 0.3',
            'W2':'normal(0, sqrt(2/8)), shape (8,1), drawn after W1','b2':'one zero',
            'matches_isolated_historical_initializer':True},
        generated_data_exactly_matches_archive=True,models=models,exact_predictor_test_rmse=exact_error,
        newly_retrained=retrain,files_written=False,
        scope='Frozen-array verification plus optional fresh in-memory reproduction. Not evidence that the historical run used this current environment.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--retrain',action='store_true')
    print(json.dumps(main(parser.parse_args().retrain),indent=2))
