import hashlib,json,os,signal,time,shutil
from pathlib import Path
root=Path('/workspace/psikoanaliz-stage-a/validation/grm_gh101_stage_a')
out=root/'results';snap=root/'user_stop_snapshot';snap.mkdir(exist_ok=False)
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def complete():
 raw=(out/'records.jsonl').read_bytes();end=raw.rfind(b'\n')+1
 return raw[:end],len(raw[:end].splitlines())
raw,n=complete();(snap/'records.prefix.jsonl').write_bytes(raw)
for name in ('metadata.json','progress.json','active_case.json'):
 p=out/name
 if p.exists():shutil.copy2(p,snap/name)
shutil.copy2('/workspace/scratch/stage-a-execution.txt',snap/'execution.txt')
save(snap/'SHA256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snap.iterdir() if p.is_file()})
save(snap/'control.json',dict(completed_when_control_installed=n,target=60,reason='Explicit user resource/quota stop, not statistical success',time=time.time()))
pid=1450
if n<60:
 fifo=out/'.user_stop_progress.pipe';os.mkfifo(fifo);os.replace(fifo,out/'progress.json')
 while n<59:
  with (out/'progress.json').open() as f:message=f.read()
  if message:
   v=json.loads(message)
   with (snap/'progress_history.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
  raw,n=complete()
 while n<60:
  time.sleep(.1);raw,n=complete()
 # Writer blocks opening progress FIFO after durable completion of60, before61.
 os.kill(pid,signal.SIGSTOP);time.sleep(.1)
 raw,n=complete()
 if n!=60 or (out/'active_case.json').exists():
  save(snap/'control_error.json',dict(completed=n,active=(out/'active_case.json').exists()))
  raise RuntimeError('Unexpected boundary; process left paused, evidence retained')
 status=Path(f'/proc/{pid}/status').read_text()
 save(snap/'boundary.json',dict(completed=n,complete_pairs=n//2,pid=pid,termination='SIGKILL only after safe durable60/FIFO barrier; no61st case',proc_status=status,time=time.time()))
 os.kill(pid,signal.SIGKILL)
 (out/'progress.json').unlink()
 save(snap/'progress_at_user_stop.json',dict(attempted=n,valid=sum(json.loads(l)['valid'] for l in raw.splitlines()),status='USER-STOPPED / INCOMPLETE',original_planned=200,resource_limited_target=60))
else:
 os.kill(pid,signal.SIGSTOP)
 save(snap/'boundary.json',dict(completed=n,excess=max(0,n-60),status='paused immediately; all excess retained',time=time.time()))
print(json.dumps({'controlled_completed':n,'initial_completed':json.loads((snap/'control.json').read_text())['completed_when_control_installed']}),flush=True)
