#!/usr/bin/env python3
"""Create a disclosed curated advisor review; never use these ratings as unbiased test accuracy."""
import csv,json,hashlib,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
import _bootstrap
from glaucoma_forecast.data.highres import preprocess_high_resolution,estimate_glaucoma_structural_maps
from glaucoma_forecast.data.preprocessing import mirror_left_eye
from glaucoma_forecast.models.residual_diffusion import build_residual_diffusion,sample_residual_trajectory
from build_sigf_locked_review_pack import tensor
import torch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/review/sigf_advisor_40_20260904'
OUT.mkdir(parents=True,exist_ok=True)
checkpoint=ROOT/'outputs/training/sigf_multitask_diffusion_v3_20260816/best.pt'
cal=json.load(open(checkpoint.parent/'change_scale.json'))
c=torch.load(checkpoint,map_location='mps',weights_only=False); cfg=c['config']
model=build_residual_diffusion(cfg['base_channels'],0,structural_channels=3,progression_head=True).to('mps')
model.load_state_dict(c.get('best_state') or c['model']);model.eval()
manifest=pd.read_csv(ROOT/'data/sigf/manifest.csv')
metrics=pd.read_csv(ROOT/'outputs/evaluation/sigf_multitask_locked_test_2x15_20260816/per_pair.csv')
score=metrics.groupby('eye_id').model_ssim.mean().to_dict()
candidates=[]
for eye,v in manifest[manifest.split.eq('test')].groupby('eye_id'):
 v=v.sort_values('time_from_baseline_years').drop_duplicates('time_from_baseline_years')
 for _,b in v.iterrows():
  later=v[(v.time_from_baseline_years-b.time_from_baseline_years>=.25)&(v.time_from_baseline_years-b.time_from_baseline_years<=5)].copy()
  if len(later)<3:continue
  picked=[]
  for target in [1,3,5]:
   idx=(later.time_from_baseline_years-b.time_from_baseline_years-target).abs().idxmin()
   picked.append(later.loc[idx]);later=later.drop(idx)
  picked.sort(key=lambda x:x.time_from_baseline_years)
  candidates.append((score.get(eye,0),eye,b,picked));break
candidates.sort(key=lambda x:(-x[0],x[1]))
chosen=[];used=set()
for item in candidates:
 if item[2].patient_id in used:continue
 chosen.append(item);used.add(item[2].patient_id)
 if len(chosen)==40:break
assert len(chosen)==40
public=[];private=[]
def prep(row):
 x=preprocess_high_resolution(str(row.image_path),cfg['image_size'],crop_fraction=cfg['disc_crop_fraction']).optic_disc
 return mirror_left_eye(x,str(row.laterality))[0]
started=time.time()
for i,(rank,eye,b,followups) in enumerate(chosen,1):
 cid=f'S{i:02d}';folder=OUT/cid;folder.mkdir(exist_ok=True)
 times=[float(r.time_from_baseline_years-b.time_from_baseline_years) for r in followups]
 base=prep(b);base.save(folder/'baseline.png')
 high=tensor(base,'mps');low=tensor(base.resize((128,128),Image.Resampling.LANCZOS),'mps')
 features=torch.cat([tensor(x.resize((128,128),Image.Resampling.BILINEAR),'mps',True) for x in estimate_glaucoma_structural_maps(base)],dim=1)
 with torch.inference_mode():
  samples=sample_residual_trajectory(model,high,low,features,times,cfg['max_supported_horizon'],cfg['max_change'],trajectories=2,diffusion_steps=15,seed=20260904+i,change_scale=cal['change_scale'])[0].mean(dim=0)
 for j,r in enumerate(followups):
  prep(r).save(folder/f'real-{j}.png')
  Image.fromarray((samples[j].permute(1,2,0).cpu().numpy().clip(0,1)*255).astype('uint8')).save(folder/f'ai-{j}.png')
 public.append({'id':cid,'years':times,'laterality':str(b.laterality)})
 private.append({'id':cid,'patient_id':str(b.patient_id),'eye_id':eye,'baseline':str(b.image_path),'baseline_label':int(b.glaucoma_label),'followups':[{'path':str(r.image_path),'years':times[j],'label':int(r.glaucoma_label)} for j,r in enumerate(followups)],'selection_score_mean_eye_test_ssim':rank,'seed':20260904+i})
 print(f'{cid}/40 ready ({time.time()-started:.0f}s)',flush=True)
metadata={'studyVersion':'sigf-advisor-curated-20260904-v1','title':'SIGF advisor review','cases':public,'curated':True,'processing':'Original optic-disc crops; left eyes mirrored consistently. No registration or color matching in the review images. AI uses the same baseline crop.'}
(OUT/'manifest.json').write_text(json.dumps(metadata,indent=2))
(OUT/'provenance.json').write_text(json.dumps({'checkpoint':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'calibration':cal,'selection':'Top mean existing test SSIM per eye; earliest baseline with three distinct followups, nearest available to 1/3/5 years; one eye per patient. Curated showcase, not an unbiased performance sample.','cases':private},indent=2))
# Contact sheets are for private research QA, not edited study evidence.
for start in range(0,40,10):
 sheet=Image.new('RGB',(1120,10*160),'#f7f5f1');d=ImageDraw.Draw(sheet)
 for line,p in enumerate(public[start:start+10]):
  folder=OUT/p['id'];d.text((5,line*160+65),p['id'],fill='black')
  for k,name in enumerate(['baseline','real-0','ai-0','real-1','ai-1','real-2','ai-2']):
   im=Image.open(folder/(name+'.png')).resize((145,145));sheet.paste(im,(75+k*148,line*160));d.text((75+k*148,line*160+146),name,fill='black')
 sheet.save(OUT/f'qa-{start//10+1}.jpg')
print('Complete',flush=True)
