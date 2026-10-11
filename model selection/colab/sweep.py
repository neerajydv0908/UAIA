"""Three fixed corrected-Adam rates; training-side selection before validation."""
import argparse
import json
from importlib.metadata import version
from pathlib import Path
from protocol import save_json, sha256, verify_bundle
from launch import bounded


def sweep(bundle, probe, output, minutes):
    import torch
    manifest=verify_bundle(bundle)
    checked=json.loads((probe/'result.json').read_text())
    expected_sources={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
    if (checked['status']!='passed' or checked['layers']!=36 or checked['bundle_sha256']!=sha256(bundle/'manifest.json')
        or checked['source_sha256']!=expected_sources or checked['gpu']!=torch.cuda.get_device_name()
        or checked['environment']!={n:version(n) for n in ('torch','transformers','peft')}):
        raise ValueError('A matching all-layer CUDA probe must pass before this sweep.')
    output.mkdir(parents=True,exist_ok=False)
    save_json(output/'plan.json',{'rates':[1e-5,2e-5,5e-5],'updates_each':150,'warmup':50,'pool_pairs':2000,
        'unique_pairs_per_rate':600,'monitor_pairs':200,'monitor_steps':[100,150],
        'labels':'hard','format':'json','length_balancing':False,'layers':36,
        'worker_minutes_each':minutes,'maximum_training_minutes':3*minutes,
        'choose_by':'minimum monitor log-loss; then lower rate, earlier step','validation_used_for_selection':False,
        'source_sha256':expected_sources,'bundle_sha256':sha256(bundle/'manifest.json')})
    candidates=[]
    for rate in (1e-5,2e-5,5e-5):
        run=output/('lr_'+str(rate))
        bounded('train.py',['--bundle',str(bundle),'--learning-rate',str(rate),'--updates','150',
                           '--wall-minutes',str(minutes)],run,minutes)
        result=json.loads((run/'result.json').read_text())
        if result['status']!='completed' or [r['step'] for r in result['monitors']] != [100,150]:
            raise ValueError('Incomplete run cannot enter sweep selection.')
        for item in result['monitors']:
            candidates.append(item|{'rate':rate,'checkpoint':str(run/f"checkpoint_{item['step']:04d}")})
    selected=min(candidates,key=lambda r:(r['log_loss'],r['rate'],r['step']))
    save_json(output/'selection.json',{'status':'completed','selected':selected,'all_candidates':candidates,
                                     'next':'Measure unchanged Qwen and this selected adapter on full validation with the FP32 head.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--probe',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--minutes-per-rate',type=int,default=120)
    args=parser.parse_args()
    if not 10<=args.minutes_per_rate<=240:parser.error('Each short-sweep worker has a 10–240 minute ceiling.')
    sweep(args.bundle,args.probe,args.output,args.minutes_per_rate)
