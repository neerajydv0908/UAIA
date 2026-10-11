"""Evaluate unchanged Qwen or one selected adapter; never open calibration or test."""
import argparse
import json
from pathlib import Path
from protocol import read_rows, save_json, sha256, verify_bundle
from train import evaluate, load_model, make_optimizer, restore_checkpoint
from report import report


def run(args):
    manifest=verify_bundle(args.bundle)
    recipe={'model':manifest['model'],'revision':manifest['revision'],'seed':20261007,'layers':36}
    if args.checkpoint:
        recipe=json.loads((args.checkpoint.parent/'plan.json').read_text())
        if recipe['bundle_sha256']!=sha256(args.bundle/'manifest.json') or recipe['format']!=args.style:
            raise ValueError('Checkpoint inputs or trained format differ; format comparisons need a separate experiment plan.')
        if recipe['source_sha256']!={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}:
            raise ValueError('Checkpoint source changed.')
    model=load_model(recipe)
    if args.checkpoint:
        optimizer=make_optimizer(model)
        restore_checkpoint(args.checkpoint,model,optimizer,recipe)
        del optimizer
    rows=read_rows(args.bundle/'validation.jsonl')
    if len(rows)!=396:raise ValueError('Incomplete validation split.')
    evaluate(model,rows,args.style,args.output)
    save_json(args.output/'evaluation_plan.json',{'model':manifest['model'],'revision':manifest['revision'],'format':args.style,
        'checkpoint':str(args.checkpoint) if args.checkpoint else None,'head':'FP32 A/B dot product',
        'bundle_sha256':sha256(args.bundle/'manifest.json'),'calibration_or_test_opened':False})
    report(args.output,args.baseline)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--baseline',type=Path);parser.add_argument('--style',choices=['json','plain'],default='json')
    run(parser.parse_args())
