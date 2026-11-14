"""Render the completed paired comparison as an exportable scientific figure."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();root=Path(a.folder)
s=json.loads((root/'summary.json').read_text());pairs=s['pairs'];formats=s['formats']
fig,axs=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
colors={'none':'#087f8c','medium':'#8b3bc2'}
for level in ['none','medium']:
 pp=[r for r in pairs if r['degrade']==level]
 axs[0].scatter([r['classic'] for r in pp],[r['hybrid'] for r in pp],c=colors[level],s=55,label='Scan limpo' if level=='none' else 'Degradação média')
lo=min([r[k] for r in pairs for k in ['classic','hybrid']])-1
hi=max([r[k] for r in pairs for k in ['classic','hybrid']])+1
axs[0].plot([lo,hi],[lo,hi],c='#999',ls='--',lw=1)
axs[0].set(xlabel='Clássico — SNR (dB)',ylabel='Híbrido — SNR (dB)',title='Comparação pareada por imagem',xlim=(lo,hi),ylim=(lo,hi))
axs[0].legend(frameon=False)
y=np.arange(len(formats));width=.35
axs[1].barh(y-width/2,[r['classic'] for r in formats],height=width,color='#087f8c',label='Clássico')
axs[1].barh(y+width/2,[r['hybrid'] for r in formats],height=width,color='#8b3bc2',label='Pesos congelados do 5º')
axs[1].set(yticks=y,yticklabels=[r['format'] for r in formats],xlabel='SNR médio das imagens (dB)',title='Média por formato (N no relatório)' )
axs[1].invert_yaxis();axs[1].legend(frameon=False)
for ax in axs:
 ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15);ax.set_axisbelow(True)
fig.suptitle('ecgdig clássico × modelo local do Kaggle: piloto de transferência',fontsize=15)
fig.text(.5,-.025,f'{len(pairs)} imagens sintéticas avaliadas • grade anônima • sem fine-tuning • sem alinhamento posterior • ausências penalizadas como zero',ha='center',fontsize=9)
fig.savefig(root/'comparison.png',dpi=180,bbox_inches='tight')
fig.savefig(root/'comparison.svg',bbox_inches='tight')
