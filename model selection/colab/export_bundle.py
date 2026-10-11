"""Local-only export: read training and validation, never calibration or final test."""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from protocol import assert_separate, read_rows, save_json, sha256, vote_probability

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from judge_v2 import DATA, messages


def encode(row, tokenizer, rubric):
    label=row['target']['preference']
    a,b=[len(row['input']['answers'][key]) for key in ('A','B')]
    encoded={'id':row['id'],'group_id':row['group_id'],'target':label,
             'strength':abs(row['target']['raw_preference_strength']),
             'shorter_won':((a<b)==(label=='A')) if label in ('A','B') and a!=b else None,
             'multi_turn':'<extra_id_1>' in row['input']['question'],
             'vote_p_a':vote_probability(row['audit_only']['all_preferences_unprocessed']), 'tokens':{}}
    for style in ('json','plain'):
        encoded['tokens'][style]=[tokenizer.apply_chat_template(messages(row,swap,style,rubric),tokenize=True,
                                  add_generation_prompt=True,return_dict=False) for swap in (False,True)]
    return encoded


def export(readiness, output):
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    from transformers import AutoTokenizer
    plan=json.loads((readiness/'plan.json').read_text())
    for name in ('train','validation'):
        if sha256(DATA/(name+'.jsonl')) != plan['source_data_sha256'][name]:
            raise ValueError('Source dataset changed: '+name)
    for name in ('train_pool.jsonl','selection_monitor.jsonl'):
        if sha256(readiness/name) != plan['files_sha256'][name]:raise ValueError('Readiness data changed.')
    pool_ids=[r['id'] for r in read_rows(readiness/'train_pool.jsonl')]
    monitor_ids=[r['id'] for r in read_rows(readiness/'selection_monitor.jsonl')]
    tokenizer=AutoTokenizer.from_pretrained(HERE.parent/'models/Qwen3-4B-Instruct-2507',local_files_only=True,trust_remote_code=False)
    if [tokenizer.encode(x,add_special_tokens=False) for x in 'AB'] != [[32],[33]]:raise ValueError('A/B token IDs changed.')
    raw_train=read_rows(DATA/'train.jsonl')
    rows=[encode(r,tokenizer,plan['rubric']) for r in raw_train]
    lookup={r['id']:r for r in rows}
    monitor=[lookup[i] for i in monitor_ids]
    monitor_groups={r['group_id'] for r in monitor}
    too_long={r['group_id'] for r in rows if max(len(ids) for pair in r['tokens'].values() for ids in pair)>2048}
    training=[r for r in rows if r['group_id'] not in monitor_groups|too_long]
    ids={r['id'] for r in training}
    if not set(pool_ids)<=ids:raise ValueError('Common-format length filter excludes original pool pairs; review before changing pool.')
    # Freeze a reproducible group order for future full-data runs.
    training.sort(key=lambda r:(sha256_text('20261007:'+r['group_id']),r['id']))
    validation=[encode(r,tokenizer,plan['rubric']) for r in read_rows(DATA/'validation.jsonl')]
    assert_separate(training,monitor,validation)
    if len(monitor)!=200 or len(validation)!=396:raise ValueError('Unexpected split size.')
    if any(len(ids)>4096 for r in monitor+validation for pair in r['tokens'].values() for ids in pair):raise ValueError('Evaluation exceeds cap.')
    output.mkdir(parents=True,exist_ok=False)
    for name,data in (('training',training),('monitor',monitor),('validation',validation)):
        with (output/(name+'.jsonl')).open('x') as stream:
            for row in data:stream.write(json.dumps(row,separators=(',',':'))+'\n')
    clear=[r for r in rows if r['target'] in ('A','B')]
    ties=[r for r in rows if r['target']=='tie_preference']
    audit={'vote_count_distribution':dict(Counter(len(r['audit_only']['all_preferences_unprocessed']) for r in raw_train)), 'raw_training_pairs':len(rows),'clear_training_pairs':len(clear),
           'clear_with_dissent':sum(0<r['vote_p_a']<1 for r in clear),
           'clear_strength_1':sum(r['strength']==1 for r in clear),
           'clear_longer_winners':sum(r['shorter_won'] is False for r in clear),
           'aggregate_ties':len(ties),'tie_vote_share_counts':dict(Counter(str(r['vote_p_a']) for r in ties)),
           'excluded_long_groups':len(too_long),'eligible_after_holdout':len(training),
           'eligible_clear_after_holdout':sum(r['target'] in ('A','B') for r in training),
           'eligible_ties_after_holdout':sum(r['target']=='tie_preference' for r in training),
           'monitor_pairs':len(monitor),'monitor_groups':len(monitor_groups),'pool_pairs':len(pool_ids),
           'validation_pairs':len(validation),'validation_clear':sum(r['target'] in ('A','B') for r in validation),
           'group_overlap':False,'calibration_and_test_opened':False}
    save_json(output/'data_audit.json',audit)
    save_json(output/'manifest.json',{'model':'Qwen/Qwen3-4B-Instruct-2507','revision':'cdbee75f17c01a7cc42f958dc650907174af0554',
        'pool2000_ids':pool_ids,'validation_groups':sorted({r['group_id'] for r in validation}),
        'monitor_groups':sorted(monitor_groups),'readiness_plan_sha256':sha256(readiness/'plan.json'),
        'rubric':plan['rubric'],'tokenizer_sha256':{p.name:sha256(p) for p in (HERE.parent/'models/Qwen3-4B-Instruct-2507').glob('*token*') if p.is_file()},
        'files_sha256':{p.name:sha256(p) for p in output.iterdir() if p.is_file()},'counts':audit,
        'data_license':'CC BY 4.0; nvidia/HelpSteer2 revision 990b2711a36180dd19d9c94b8627844866f8982a',
        'notice':'Tokenized copies of public preference data; annotator prose and expected labels never enter model prompts. Calibration/test excluded.'})
    print(json.dumps(audit,indent=2))


def sha256_text(value):
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('readiness',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();export(args.readiness,args.output)
