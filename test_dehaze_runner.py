#!/usr/bin/env python3
"""Run targeted dehaze() calls for quick verification (images 0006-0010)."""
import runpy, os, glob
ROOT = os.path.abspath(os.path.dirname(__file__))
E = runpy.run_path(os.path.join(ROOT, 'Eval.py'))
imgs = [os.path.join(ROOT, 'dataset', 'Haze4K', 'test', 'hazy_school', f'{i:04d}.jpg') for i in range(6,11)]
outdir = os.path.join(ROOT, 'outputs', 'clip_dehaze', 'test_run2')
os.makedirs(outdir, exist_ok=True)
print('CLIP module present?', bool(E.get('clip_mod')))
print('CLIP model loaded?', bool(E.get('clip_model_global')))
for im in imgs:
    print('Processing', im)
    try:
        E['dehaze'](E.get('model'), im, outdir)
    except Exception as ex:
        print('Error for', im, ex)

print('\nGenerated files in test dir:')
for p in sorted(glob.glob(os.path.join(outdir, '*'))):
    print('  ', os.path.basename(p))

cd = os.path.join(outdir, 'clip_descriptions.txt')
if os.path.exists(cd):
    print('\n--- clip_descriptions.txt ---')
    with open(cd, 'r', encoding='utf-8') as f:
        for l in f.readlines()[:20]:
            print(l.strip())
else:
    print('\nNo clip_descriptions.txt generated in test run')
