"""Two CUDA updates, a long-prompt backward pass, and exact checkpoint restart."""
import argparse
import gc
import json
import time
from importlib.metadata import version
from pathlib import Path
from protocol import read_rows, save_json, sha256, verify_bundle
from train import capture_rng, adapter_state, decision_scores, evaluate, load_model, loss_for, make_optimizer, restore_checkpoint, save_checkpoint, train_update


def probe(bundle, output, layers):
    import torch
    manifest=verify_bundle(bundle)
    rows=read_rows(bundle/'training.jsonl')
    lookup={r['id']:r for r in rows}
    pool=[lookup[i] for i in manifest['pool2000_ids']]
    ordered=sorted(pool,key=lambda r:max(map(len,r['tokens']['json'])))
    samples=[ordered[int((i+.5)*len(ordered)/8)] for i in range(8)]
    recipe={'model':manifest['model'],'revision':manifest['revision'],'seed':20261007,'layers':layers,
            'learning_rate':2e-5,'warmup':50,'updates':150,'format':'json','labels':'hard'}
    output.mkdir(parents=True,exist_ok=False)
    save_json(output/'plan.json',recipe)
    tick=time.monotonic()
    model=load_model(recipe);optimizer=make_optimizer(model)
    weights={r['id']:1. for r in samples}
    entries=[train_update(model,optimizer,samples[:4],weights,recipe,0)]
    save_checkpoint(output/'checkpoint_0001',model,optimizer,1,recipe)
    evaluate(model,samples[:2],'json',output/'evaluation_between_updates')
    entries.append(train_update(model,optimizer,samples[4:],weights,recipe,1))
    expected_rng=capture_rng()
    expected=adapter_state(model)
    expected_opt={i:{k:v.detach().cpu().clone() for k,v in state.items()} for i,state in optimizer.state_dict()['state'].items()}
    # Exercise the longest exported input, even if it is absent from the 2,000-pair pool.
    longest=max(rows,key=lambda r:max(len(ids) for pair in r['tokens'].values() for ids in pair))
    ids=max((ids for pair in longest['tokens'].values() for ids in pair),key=len)
    optimizer.zero_grad(set_to_none=True)
    value=loss_for(decision_scores(model,torch.tensor([ids],device='cuda')),.5)
    value.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    del value,model,optimizer
    gc.collect();torch.cuda.empty_cache()
    restarted=load_model(recipe);optimizer=make_optimizer(restarted)
    restore_checkpoint(output/'checkpoint_0001',restarted,optimizer,recipe)
    repeated=train_update(restarted,optimizer,samples[4:],weights,recipe,1)
    actual=adapter_state(restarted)
    exact_weights=all(torch.equal(value,actual[name]) for name,value in expected.items())
    exact_optimizer=all(torch.equal(value,optimizer.state_dict()['state'][i][key].cpu()) for i,state in expected_opt.items() for key,value in state.items())
    actual_rng=capture_rng()
    exact_rng=(torch.equal(expected_rng['cpu'],actual_rng['cpu']) and expected_rng['python']==actual_rng['python']
               and len(expected_rng['cuda'])==len(actual_rng['cuda'])
               and all(torch.equal(a,b) for a,b in zip(expected_rng['cuda'],actual_rng['cuda'])))
    if not exact_rng or not exact_weights or not exact_optimizer or entries[1]['mean_unweighted_loss']!=repeated['mean_unweighted_loss']:
        raise ValueError('CUDA restart changed weights, Adam state, or dropout-dependent loss.')
    save_json(output/'result.json',{'status':'passed','layers':layers,'updates':entries,'longest_tokens':len(ids),
        'exact_restart_weights_optimizer_random_loss':True,'seconds':time.monotonic()-tick,
        'peak_cuda_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
        'peak_cuda_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
        'gpu':torch.cuda.get_device_name(),'torch_cuda':torch.version.cuda,
        'environment':{n:version(n) for n in ('torch','transformers','peft')},
        'bundle_sha256':sha256(bundle/'manifest.json'),
        'source_sha256':{p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
        'interpretation':'Readiness only; no quality claim. Monitor and full validation still need scoring.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--layers',type=int,choices=[16,36],default=36)
    args=parser.parse_args();probe(args.bundle,args.output,args.layers)
