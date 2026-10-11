"""Bounded CUDA/PEFT training; the notebook starts with a probe, not a long run."""
import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
from protocol import assert_separate, basic_metrics, coverage_curve, epoch_batches, learning_rate, read_rows, save_json, sha256, target_probability, training_weights, verify_bundle


def decision_scores(model, tokens):
    import torch
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    hidden = base.model(input_ids=tokens, use_cache=False, return_dict=True).last_hidden_state[:, -1, :]
    weight = base.get_output_embeddings().weight[[32, 33]]
    with torch.autocast(device_type=tokens.device.type, enabled=False):
        return hidden.float() @ weight.float().T


def make_optimizer(model):
    import torch
    return torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-5, betas=(.9,.999), eps=1e-8, weight_decay=0, foreach=False, fused=False)


def loss_for(scores, probability):
    import torch
    target = torch.tensor([[probability, 1-probability]], dtype=torch.float32, device=scores.device)
    return -(target * scores.log_softmax(-1)).sum()


def train_update(model, optimizer, rows, weights, recipe, index):
    import torch
    if not 1 <= len(rows) <= 4:
        raise ValueError('Each update keeps both orders of up to four pairs together.')
    presentations = 2 * len(rows)
    optimizer.zero_grad(set_to_none=True)
    rate = learning_rate(index, recipe['learning_rate'], recipe['warmup'], recipe['updates'])
    for group in optimizer.param_groups:
        group['lr'] = rate
    tick = time.monotonic()
    readings = []
    for i, row in enumerate(rows):
        for swapped in ((False, True) if i % 2 == 0 else (True, False)):
            tokens = row['tokens'][recipe['format']][int(swapped)]
            if not 0 < len(tokens) <= 2048:
                raise ValueError('Training prompt exceeds the cap; truncation is forbidden.')
            probability = target_probability(row, recipe['labels'], swapped)
            value = loss_for(decision_scores(model, torch.tensor([tokens], device=next(model.parameters()).device)), probability)
            if not torch.isfinite(value):
                raise ValueError('Non-finite training loss.')
            (value * weights[row['id']] / presentations).backward()
            readings.append(float(value.detach()))
    parameters = [p for p in model.parameters() if p.requires_grad]
    raw_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0, error_if_nonfinite=True)
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in parameters):
        raise ValueError('Non-finite adapter weights after update.')
    return {'step':index+1, 'learning_rate':rate, 'pairs':len(rows), 'mean_unweighted_loss':sum(readings)/presentations,
            'raw_gradient_norm':float(raw_norm), 'clip_norm':1.0, 'seconds':time.monotonic()-tick}


def capture_rng():
    import torch
    return {'cpu':torch.get_rng_state(), 'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], 'python':random.getstate()}


def restore_rng(state):
    import torch
    torch.set_rng_state(state['cpu'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    random.setstate(state['python'])


def adapter_state(model):
    return {name:p.detach().cpu().clone() for name,p in model.named_parameters() if p.requires_grad}


def save_checkpoint(folder, model, optimizer, step, recipe):
    import torch
    folder = Path(folder)
    temporary = folder.with_name(folder.name+'.incomplete')
    if folder.exists():
        raise ValueError('Never overwrite an existing checkpoint.')
    temporary.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(temporary/'adapter', safe_serialization=True)
    torch.save({'adapter':adapter_state(model), 'optimizer':optimizer.state_dict(), 'rng':capture_rng(),
                'step':step, 'recipe':recipe}, temporary/'resume.pt')
    save_json(temporary/'receipt.json', {'files':{str(p.relative_to(temporary)):sha256(p) for p in temporary.rglob('*') if p.is_file()}})
    temporary.rename(folder)


def restore_checkpoint(folder, model, optimizer, recipe):
    import torch
    folder = Path(folder)
    for name,expected in json.loads((folder/'receipt.json').read_text())['files'].items():
        if sha256(folder/name) != expected:
            raise ValueError('Changed checkpoint file: '+name)
    saved = torch.load(folder/'resume.pt', map_location='cpu', weights_only=True)
    if saved['recipe'] != recipe:
        raise ValueError('Resume recipe or input/source fingerprint changed.')
    parameters = dict(model.named_parameters())
    expected = {name for name,p in parameters.items() if p.requires_grad}
    if expected != saved['adapter'].keys():
        raise ValueError('Checkpoint adapter structure changed.')
    with torch.no_grad():
        for name,value in saved['adapter'].items():
            parameters[name].copy_(value)
    optimizer.load_state_dict(saved['optimizer'])
    restore_rng(saved['rng'])
    return saved['step']


def evaluate(model, rows, style, destination):
    import torch
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    training, rng = model.training, capture_rng()
    model.eval()
    results = []
    try:
        with torch.no_grad(), (destination/'scores.jsonl').open('x') as stream:
            for row in rows:
                probabilities = []
                for swapped in (False, True):
                    ids = row['tokens'][style][int(swapped)]
                    if not 0 < len(ids) <= 4096:
                        raise ValueError('Evaluation input exceeds the cap.')
                    scores = decision_scores(model, torch.tensor([ids], device=next(model.parameters()).device))
                    p = float(scores.softmax(-1)[0,0])
                    probabilities.append(1-p if swapped else p)
                    stream.write(json.dumps({'id':row['id'],'swapped':swapped,'logits':scores[0].tolist(),'p_display_a':p})+'\n')
                    stream.flush()
                results.append({k:row[k] for k in ('id','group_id','target','strength','shorter_won','multi_turn')} | {'p_a':sum(probabilities)/2, 'original':probabilities[0], 'swapped':probabilities[1]})
                save_json(destination/'progress.json', {'pairs':len(results),'planned':len(rows)})
        metrics = basic_metrics(results)
        save_json(destination/'pairs.json', results)
        save_json(destination/'metrics.json', metrics)
        save_json(destination/'coverage.json', coverage_curve(results))
        return metrics
    finally:
        model.train(training)
        restore_rng(rng)


def load_model(recipe, tiny=False):
    import torch
    from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(recipe['seed'])
    random.seed(recipe['seed'])
    if tiny:
        base = Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                 num_attention_heads=2, num_key_value_heads=1, head_dim=8, tie_word_embeddings=True, attention_dropout=0.0))
    else:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('Select a BF16-capable Colab CUDA GPU, such as L4/A100. CPU/MPS/FP16 fallback is not this experiment.')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        base = AutoModelForCausalLM.from_pretrained(recipe['model'], revision=recipe['revision'], dtype=torch.bfloat16,
                     attn_implementation='sdpa', trust_remote_code=False).to('cuda')
        if base.config.num_hidden_layers != 36 or base.config.hidden_size != 2560 or not base.config.tie_word_embeddings:
            raise ValueError('Unexpected Qwen architecture.')
    base.config.use_cache = False
    layers = list(range(base.config.num_hidden_layers)) if tiny else list(range(36-recipe['layers'],36))
    lora = LoraConfig(r=16, lora_alpha=320, lora_dropout=.05, bias='none', task_type='CAUSAL_LM',
                     target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
                     layers_to_transform=layers, use_rslora=False, init_lora_weights=True)
    model = get_peft_model(base, lora)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    parameters = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if len(parameters) != len(layers)*7*2 or any('lora_' not in n for n,p in parameters):
        raise ValueError('Unexpected trainable parameter set.')
    for _,p in parameters:
        p.data = p.data.float()
    model.train()
    return model


def execute(args):
    import torch
    from importlib.metadata import version
    folder, output = Path(args.bundle), Path(args.output)
    manifest = verify_bundle(folder)
    training = read_rows(folder/'training.jsonl')
    monitor = read_rows(folder/'monitor.jsonl')
    # Only group identities, not validation contents, are needed during training.
    validation_groups = manifest['validation_groups']
    assert_separate(training, monitor, [{'id':g,'group_id':g} for g in validation_groups])
    if args.pool == 'pool2000':
        ids = manifest['pool2000_ids']
        lookup = {r['id']:r for r in training}
        training = [lookup[i] for i in ids]
    elif args.pool == 'all_clear':
        training = [r for r in training if r['target'] in ('A','B')]
    if args.max_pairs is not None:
        if not 1 <= args.max_pairs <= len(training):
            raise ValueError('Requested pair limit exceeds the eligible pool.')
        training = training[:args.max_pairs]
    if args.labels != 'votes' and any(r['target']=='tie_preference' for r in training):
        raise ValueError('Use vote targets when including tie-labelled pairs.')
    if args.updates > math.ceil(len(training)/4)*args.epochs:
        raise ValueError('Requested updates exceed the declared number of data passes.')
    if not 0 <= args.minimum_epochs <= args.epochs or args.updates < math.ceil(len(training)/4)*args.minimum_epochs:
        raise ValueError('The minimum complete epochs must fit within the declared update budget.')
    weights = training_weights(training, args.balance_length, args.weight_strength)
    batches = epoch_batches(training, args.epochs, args.seed, args.shuffle_epochs)
    if args.resume:
        recipe = json.loads((output/'plan.json').read_text())
    else:
        output.mkdir(parents=True, exist_ok=False)
        recipe = vars(args).copy()
        recipe.update(model=manifest['model'],revision=manifest['revision'],format=args.style,
                      bundle_sha256=sha256(folder/'manifest.json'),source_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
                      environment={n:version(n) for n in ('torch','transformers','peft')},
                      runtime={'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'},
                      adam_bias_correction=True,clip_norm=1.0,
                      training_pair_ids=[r['id'] for r in training],
                      planned_pair_batches=[[r['id'] for r in batch] for batch in batches])
        recipe.pop('resume')
        save_json(output/'plan.json',recipe)
    if recipe['bundle_sha256'] != sha256(folder/'manifest.json') or recipe['source_sha256'] != {p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}:
        raise ValueError('Recipe inputs changed.')
    current = vars(args).copy(); current.pop('resume')
    if any(recipe[k] != v for k,v in current.items()):
        raise ValueError('CLI settings differ from the saved plan.')
    if recipe['environment'] != {n:version(n) for n in ('torch','transformers','peft')}:
        raise ValueError('Resume environment changed.')
    if recipe['runtime'] != {'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}:
        raise ValueError('Resume CUDA runtime or GPU type changed.')
    if args.resume:
        resume_folder = Path(args.resume).resolve()
        checkpoints = sorted(output.glob('checkpoint_[0-9][0-9][0-9][0-9]'))
        if not checkpoints or resume_folder != checkpoints[-1].resolve() or (output/'result.json').exists():
            raise ValueError('Explicit resume must use the latest complete checkpoint of an unfinished run.')
    model = load_model(recipe)
    optimizer = make_optimizer(model)
    start = restore_checkpoint(args.resume,model,optimizer,recipe) if args.resume else 0
    if not 0 <= start < args.updates:
        raise ValueError('Invalid checkpoint cursor.')
    if args.resume:
        archive = output / ('interrupted_' + str(time.time_ns()))
        archive.mkdir()
        for name in ('updates.jsonl','progress.json','failure.json'):
            if (output/name).exists(): shutil.move(str(output/name),archive/name)
        for path in output.glob('checkpoint_*.incomplete'):
            shutil.move(str(path),archive/path.name)
        for path in output.glob('monitor_*'):
            if not (path/'metrics.json').exists(): shutil.move(str(path),archive/path.name)
    if recipe['planned_pair_batches'] != [[r['id'] for r in batch] for batch in batches]:
        raise ValueError('Recorded epoch order changed.')
    started = time.monotonic()
    used = json.loads((output/'elapsed.json').read_text())['seconds'] if (output/'elapsed.json').exists() else 0
    monitors = []
    for existing in sorted(output.glob('monitor_*/metrics.json')):
        monitors.append({'step':int(existing.parent.name.split('_')[-1]),**json.loads(existing.read_text())})
    if start and (start % args.monitor_every == 0) and not any(r['step']==start for r in monitors):
        measured = evaluate(model,monitor,args.style,output/f'monitor_{start:04d}')
        monitors.append({'step':start,**measured})
    with (output/'updates.jsonl').open('x') as stream:
        for index in range(start,args.updates):
            elapsed = used+time.monotonic()-started
            save_json(output/'elapsed.json',{'seconds':elapsed})
            if elapsed >= args.wall_minutes*60:
                raise TimeoutError('The declared worker-time budget is exhausted; no automatic retry.')
            entry = train_update(model,optimizer,batches[index],weights,recipe,index)
            entry['epoch'] = index // math.ceil(len(training)/4) + 1
            stream.write(json.dumps(entry)+'\n');stream.flush()
            save_json(output/'progress.json',entry)
            print(json.dumps(entry),flush=True)
            step = index+1
            if step%args.checkpoint_every==0 or step==args.updates:
                save_checkpoint(output/f'checkpoint_{step:04d}',model,optimizer,step,recipe)
            if step%args.monitor_every==0 or step==args.updates:
                if not (output/f'checkpoint_{step:04d}').exists():
                    save_checkpoint(output/f'checkpoint_{step:04d}',model,optimizer,step,recipe)
                measured = evaluate(model,monitor,args.style,output/f'monitor_{step:04d}')
                monitors.append({'step':step,**measured})
                if args.patience and len(monitors)>args.patience and step >= math.ceil(len(training)/4)*args.minimum_epochs:
                    previous_best=min(r['log_loss'] for r in monitors[:-args.patience])
                    recent_best=min(r['log_loss'] for r in monitors[-args.patience:])
                    if recent_best>previous_best-args.minimum_improvement:
                        break
    elapsed = used+time.monotonic()-started
    save_json(output/'elapsed.json',{'seconds':elapsed})
    if elapsed > args.wall_minutes*60:
        raise TimeoutError('Budget expired; incomplete runs cannot enter selection.')
    best = min(monitors,key=lambda r:(r['log_loss'],r['step']))
    save_json(output/'result.json',{'status':'completed','best':best,'monitors':monitors,'seconds':elapsed,
              'completed_updates':step,'stopped_early':step<args.updates,
              'peak_cuda_gib':torch.cuda.max_memory_allocated()/1024**3,'validation_opened':False})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--learning-rate',type=float,required=True)
    parser.add_argument('--updates',type=int,default=150)
    parser.add_argument('--warmup',type=int,default=50);parser.add_argument('--epochs',type=int,choices=[1,2],default=1)
    parser.add_argument('--labels',choices=['hard','smooth','votes'],default='hard')
    parser.add_argument('--style',choices=['json','plain'],default='json')
    parser.add_argument('--pool',choices=['pool2000','all_clear','all_votes'],default='pool2000')
    parser.add_argument('--layers',type=int,choices=[16,36],default=36)
    parser.add_argument('--balance-length',action='store_true');parser.add_argument('--resume')
    parser.add_argument('--weight-strength',action='store_true')
    parser.add_argument('--max-pairs',type=int)
    parser.add_argument('--seed',type=int,default=20261007)
    parser.add_argument('--fixed-epoch-order',dest='shuffle_epochs',action='store_false')
    parser.add_argument('--monitor-every',type=int,default=100)
    parser.add_argument('--checkpoint-every',type=int,default=25)
    parser.add_argument('--patience',type=int,default=0)
    parser.add_argument('--minimum-epochs',type=int,default=0)
    parser.add_argument('--minimum-improvement',type=float,default=.002)
    parser.add_argument('--wall-minutes',type=int,default=120)
    args=parser.parse_args()
    if not 10 <= args.wall_minutes <= 720 or args.monitor_every<1 or args.checkpoint_every<1 or not 3 < args.updates <= 4000 or not 1 < args.warmup < args.updates-1 or args.patience<0 or args.minimum_improvement<0:
        parser.error('Use a bounded 10–720 minute budget and a positive monitor interval.')
    try:
        execute(args)
    except Exception as error:
        output=Path(args.output)
        if output.exists() and not (output/'result.json').exists() and not (output/'failure.json').exists():
            save_json(output/'failure.json',{'type':type(error).__name__,'message':str(error)})
        raise


if __name__=='__main__':
    main()
