"""External wall-clock limit; failed workers are recorded and never retried."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from protocol import save_json


def bounded(script, arguments, output, minutes):
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    # Keep supervisor records outside the worker folder, which workers create once.
    stamp=str(time.time_ns())
    log=output.parent/(output.name+'.'+stamp+'.log')
    receipt=output.parent/(output.name+'.'+stamp+'.execution.json')
    previous=json.loads((output/'elapsed.json').read_text())['seconds'] if (output/'elapsed.json').exists() else 0
    remaining=minutes*60-previous
    if remaining<=0:raise ValueError('Cumulative worker budget is already exhausted.')
    command=[sys.executable,str(Path(__file__).parent/script),'--output',str(output),*arguments]
    tick=time.monotonic()
    with log.open('x') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
        save_json(receipt,{'status':'running','pid':process.pid,'command':command,'wall_minutes':minutes,'log':str(log)})
        try:
            code=process.wait(timeout=remaining)
            status='completed' if code==0 else 'failed'
        except subprocess.TimeoutExpired:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            code=process.returncode;status='timeout'
    elapsed=previous+time.monotonic()-tick
    save_json(receipt,{'status':status,'returncode':code,'cumulative_seconds':elapsed,'log':str(log),'automatic_retries':0})
    if output.exists():save_json(output/'elapsed.json',{'seconds':elapsed})
    print(json.dumps({'status':status,'receipt':str(receipt),'log':str(log)}),flush=True)
    if status!='completed':raise RuntimeError('Worker '+status+'; inspect its log and receipt before a manual restart.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--minutes',type=int,required=True)
    parser.add_argument('--output',required=True);parser.add_argument('mode',choices=['probe','train','evaluate'])
    args,remaining=parser.parse_known_args()
    if not 1<=args.minutes<=720:parser.error('Budget must be between 1 and 720 minutes.')
    bounded(args.mode+'.py',remaining,args.output,args.minutes)
