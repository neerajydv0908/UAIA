"""Full 396-pair reports with the unchanged development screen and paired uncertainty."""
import argparse
import json
from pathlib import Path
from metrics import choice, compare, format_passes, measure
from protocol import coverage_curve, save_json

SETTINGS={'bootstrap_replicates':10000,'bootstrap_seed':20261007,'minimum_accuracy_gain':.03,
          'minimum_logloss_reduction':.02,'familywise_alpha':.05}


def checked_pairs(path):
    rows=json.loads(Path(path).read_text())
    if len(rows)!=396 or sum(r['target'] in ('A','B') for r in rows)!=311 or len({r['id'] for r in rows})!=396:
        raise ValueError('Require every validation pair, including all 85 ties.')
    for r in rows:
        if not all(0<=r[k]<=1 for k in ('p_a','original','swapped')) or abs(r['p_a']-(r['original']+r['swapped'])/2)>1e-12:
            raise ValueError('Malformed two-order probabilities.')
    return [r|{'averaged':r['p_a'],'status':'completed'} for r in rows]


def report(folder, baseline=None):
    rows=checked_pairs(folder/'pairs.json')
    clear=[r for r in rows if r['target'] in ('A','B')];ties=[r for r in rows if r['target']=='tie_preference']
    result={'views':{field:measure(clear,field,SETTINGS) for field in ('original','swapped','averaged')},
            'strength':{str(k):measure([r for r in clear if r['strength']==k],'averaged',SETTINGS) for k in (1,2,3)},
            'shorter_winner':measure([r for r in clear if r['shorter_won'] is True],'averaged',SETTINGS),
            'turns':{str(k):measure([r for r in clear if r['multi_turn'] is k],'averaged',SETTINGS) for k in (False,True)},
            'order_agreement':sum(choice(r['original'])==choice(r['swapped']) for r in rows)/len(rows),
            'ties':{'pairs':len(ties),'mean_p_a':sum(r['p_a'] for r in ties)/len(ties),
                    'mean_distance_from_half':sum(abs(r['p_a']-.5) for r in ties)/len(ties),
                    'bands':[{'margin':b,'tie_recall':sum(abs(r['p_a']-.5)<=b for r in ties)/len(ties),
                              'clear_coverage':sum(abs(r['p_a']-.5)>b for r in clear)/len(clear)} for b in (0,.025,.05,.075,.1,.15,.2)]},
            'coverage':coverage_curve(rows),'calibration':'not_run; margins and coverage are development diagnostics',
            'interpretation':'95% question-group bootstrap intervals; repeated development comparisons remain exploratory.'}
    if baseline:
        before=checked_pairs(baseline/'pairs.json')
        comparison=compare(before,rows,SETTINGS)
        comparison['mcnemar']['holm_adjusted_p']=comparison['mcnemar']['exact_two_sided_p']
        result['paired_comparison']=comparison
        result['development_screen']=format_passes(comparison,SETTINGS)
        result['comparison_notice']='One primary comparison for the selected corrected-rate checkpoint. Multiple later ablations require a new comparison-family plan.'
    save_json(folder/'full_report.json',result)
    print(json.dumps({'matches':result['views']['averaged']['matches'], 'accuracy':result['views']['averaged']['metrics']['accuracy'],
                      'development_screen':result.get('development_screen')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);parser.add_argument('--baseline',type=Path)
    args=parser.parse_args();report(args.folder,args.baseline)
