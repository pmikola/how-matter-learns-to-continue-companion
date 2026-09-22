"""Deterministic teaching figures. No confirmatory Alpha-Omega campaign is run.

All numerical panels are exact formula evaluations or explicitly seeded toy samples.
The E3 panel transcribes a published local report, it does not recompute its test.
"""
from pathlib import Path
import json, math, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/investigation-v24/figures'; OUT.mkdir(parents=True,exist_ok=True)
BLUE='#0072B2'; ORANGE='#D55E00'; GRAY='#73777B'; INK='#202326'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
 'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,
 'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':GRAY,
 'text.color':INK,'axes.labelcolor':INK,'pdf.fonttype':42,'savefig.facecolor':'white'})
data={}; checks={}
def save(fig,name,record):
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight',pad_inches=.12)
    fig.savefig(OUT/f'{name}.png',dpi=180,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    data[name]=record
def pair(height=3.35):return plt.subplots(1,2,figsize=(7.3,height),layout='constrained')

# 15: Four different questions about the same temperature reading.
t=np.linspace(0,60,241)
fig,ax=plt.subplots(2,2,figsize=(7.3,4.7),layout='constrained',sharex=True,sharey=True)
traces=[350+0*t,300+50*np.exp(-t/20),300+50*np.exp(-t/40),300+30*np.exp(-t/20)]
titles=['Keep both contacts','Remove the hot contact','Also double heat capacity','Instead start at 330 K']
for a,y,title in zip(ax.flat,traces,titles):
    a.plot(t,y,color=BLUE,lw=2);a.set_title(title);a.set_ylim(298,354)
    a.set_xlabel('Time after decision (s)');a.set_ylabel('Temperature (K)')
save(fig,'why-comparisons',{'t':t.tolist(),'temperature':[x.tolist() for x in traces],
 'status':'analytic lumped heat model, not measurements','C_J_per_K':[20,20,40,20],
 'remaining_cold_contact_W_per_K':1})

# 16: Exact right-moving compact pulse of the continuum wave equation.
x=np.linspace(-2,8,501); times=np.linspace(0,3,151)
def pulse(z):return np.where(np.abs(z)<1,.01*np.cos(np.pi*z/2)**2,0)
fig,ax=pair(3.7)
for ti,c,ls in [(0,GRAY,':'),(1,BLUE,'-'),(2,ORANGE,'--')]:
    ax[0].plot(x,1000*pulse(x-2*ti),color=c,ls=ls,lw=2,label=f'{ti} s')
ax[0].set(xlabel='Position (m)',ylabel='Displacement (mm)',title='The shape travels')
ax[0].legend(frameon=False)
z=1000*pulse(x[None,:]-2*times[:,None])
im=ax[1].pcolormesh(x,times,z,cmap='Blues',shading='auto',vmin=0,vmax=10,rasterized=True)
ax[1].plot(2*times-1,times,color=ORANGE,ls='--');ax[1].plot(2*times+1,times,color=ORANGE,ls='--')
ax[1].set(xlabel='Position (m)',ylabel='Time (s)',title='A specified finite front')
fig.colorbar(im,ax=ax[1],label='Displacement (mm)',shrink=.8)
save(fig,'field-pulse',{'x_m':x.tolist(),'t_s':times.tolist(),'speed_m_per_s':2,
 'amplitude_m':.01,'half_width_m':1,'status':'exact continuum travelling-wave solution'})

# 17: Two masses, two wall springs, one coupling spring.
t=np.linspace(0,22,1001); w=np.sqrt(2)
q1=.5*(np.cos(t)+np.cos(w*t));q2=.5*(np.cos(t)-np.cos(w*t))
v1=-.5*(np.sin(t)+w*np.sin(w*t));v2=-.5*(np.sin(t)-w*np.sin(w*t))
kin=.5*(v1*v1+v2*v2); wall=.5*(q1*q1+q2*q2); coupling=.25*(q1-q2)**2
checks['oscillator_energy_max_error']=float(np.max(np.abs(kin+wall+coupling-.75)))
assert checks['oscillator_energy_max_error']<1e-12
fig,ax=pair()
ax[0].plot(t,q1,color=BLUE,label='Mass 1');ax[0].plot(t,q2,color=ORANGE,ls='--',label='Mass 2')
ax[0].set(xlabel='Time (s)',ylabel='Displacement (m)',title='Motion passes between masses');ax[0].legend(frameon=False)
for y,label,col,ls in [(kin,'Kinetic',BLUE,'-'),(wall,'Wall springs',ORANGE,'--'),(coupling,'Coupling spring',GRAY,':')]:
    ax[1].plot(t,y,color=col,ls=ls,label=label)
ax[1].axhline(.75,color=INK,lw=1.8,label='Total: 0.75 J')
ax[1].set(xlabel='Time (s)',ylabel='Energy (J)',title='Include the interaction energy');ax[1].legend(frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.22),ncol=2)
save(fig,'conserved-energy',{'t_s':t.tolist(),'q1_m':q1.tolist(),'q2_m':q2.tolist(),
 'kinetic_J':kin.tolist(),'wall_J':wall.tolist(),'coupling_J':coupling.tolist(),'total_J':.75})

# 19: Events in two inertial frames, c = one light-second per second.
events=np.array([[0,0],[1.8,3],[3,1.]]) # x,t
beta=.6;gamma=1.25
transform=lambda p: np.array([gamma*(p[:,0]-beta*p[:,1]),gamma*(p[:,1]-beta*p[:,0])]).T
ep=transform(events)
checks['interval_invariance']=float(np.max(np.abs(events[:,1]**2-events[:,0]**2-(ep[:,1]**2-ep[:,0]**2))))
assert checks['interval_invariance']<1e-12
fig,ax=pair(3.8)
for a,ev,title in [(ax[0],events,'Frame A'),(ax[1],ep,'Frame B: moving at 0.6 c')]:
    u=np.linspace(-1,4,101)
    a.fill_betweenx(u,-np.abs(u),np.abs(u),color=BLUE,alpha=.08)
    a.plot(u,u,color=GRAY,ls=':');a.plot(-u,u,color=GRAY,ls=':')
    a.plot(ev[:2,0],ev[:2,1],color=BLUE,lw=2)
    for j,(xx,tt) in enumerate(ev):
        a.scatter(xx,tt,color=ORANGE if j==2 else BLUE,zorder=4)
        a.annotate(['Emission','Reception','Separate event'][j],(xx,tt),xytext=(5,7),textcoords='offset points',fontsize=8)
    a.set(xlim=(-3.5,4.5),ylim=(-1.4,4),xlabel='Position (light-seconds)',ylabel='Time (s)',title=title)
save(fig,'spacetime-events',{'events_x_lightseconds_t_seconds':events.tolist(),'transformed':ep.tolist(),
 'beta':beta,'proper_time_reception_s':2.4,'status':'exact special-relativity coordinates'})

# 20: Frozen Newtonian tidal tensor as a local, weak-field teaching limit.
t=np.linspace(0,5,201); kr=.002;kt=.001
rad=np.cosh(np.sqrt(kr)*t);tan=np.cos(np.sqrt(kt)*t)
fig,ax=pair()
angle=np.linspace(0,2*np.pi,201)
ax[0].plot(np.cos(angle),np.sin(angle),color=GRAY,ls=':',label='Initial circle')
ax[0].plot(rad[-1]*np.cos(angle),tan[-1]*np.sin(angle),color=BLUE,label='After 5 s')
ax[0].set(xlabel='Radial separation / initial radius',ylabel='Transverse separation / initial radius',title='Nearby falls separate differently',aspect='equal');ax[0].legend(frameon=False,fontsize=8)
ax[1].plot(t,rad,color=BLUE,label='Radial');ax[1].plot(t,tan,color=ORANGE,ls='--',label='Transverse')
ax[1].set(xlabel='Time (s)',ylabel='Separation / initial separation',title='A local tidal approximation');ax[1].legend(frameon=False)
save(fig,'tidal-separation',{'t_s':t.tolist(),'radial_ratio':rad.tolist(),'transverse_ratio':tan.tolist(),
 'mu_m3_per_s2':1,'reference_radius_m':10,'status':'constant-coefficient local Newtonian tidal approximation, not full GR evolution'})

# 21: Exactly reversible 12-site colour ring. Explicit inverse, not entropy dynamics.
mask=np.zeros(12,dtype=int);mask[[0,3,7]]=1
state=np.zeros(12,dtype=int);history=[state.copy()]
for step in range(24):state=np.roll(state,1)^mask;history.append(state.copy())
hist=np.array(history);n=hist.sum(axis=1);S=np.array([math.log(math.comb(12,int(k))) for k in n])
back=hist[10].copy()
for step in range(10):back=np.roll(back^mask,-1)
assert np.array_equal(back,hist[0]);assert np.array_equal(hist[-1],hist[0])
fig,ax=pair(3.7)
ax[0].imshow(hist[:13],aspect='auto',cmap=ListedColormap([BLUE,ORANGE]),interpolation='nearest',extent=(-.5,11.5,12.5,-.5))
ax[0].set(xlabel='Site',ylabel='Update',title='Each microscopic update is invertible');ax[0].set_xticks([0,3,7,11])
ax[1].plot(np.arange(25),S,'o-',color=BLUE,ms=3,label='Forward rule')
ax[1].plot(np.arange(10,21),S[10::-1],color=ORANGE,ls='--',lw=2,label='Inverse from update 10')
ax[1].set(xlabel='Displayed update count',ylabel='Log number of matching colourings',title='Coarse entropy need not rise forever');ax[1].legend(frameon=False,fontsize=8)
save(fig,'reversible-record',{'states':hist.tolist(),'count_orange':n.tolist(),'log_multiplicity':S.tolist(),
 'flippers':[0,3,7],'inverse_start':10,'status':'exact finite invertible toy, not a thermal gas'})

# 24: One visible and one hidden component, exact solution.
t=np.linspace(0,5,201);q=np.exp(-t)-np.exp(-2*t)
fig,ax=pair()
for sign,col,ls in [(1,BLUE,'-'),(-1,ORANGE,'--')]:
    ax[0].plot(t,sign*q,color=col,ls=ls,label=f'Hidden r(0) = {sign:+d}')
    ax[1].plot(t,sign*np.exp(-2*t),color=col,ls=ls,label=f'r(0) = {sign:+d}')
ax[0].scatter([0],[0],color=INK,zorder=4);ax[0].set(xlabel='Time (chosen unit)',ylabel='Visible q (dimensionless)',title='Same present, different futures');ax[0].legend(frameon=False)
ax[1].set(xlabel='Time (chosen unit)',ylabel='Hidden r (dimensionless)',title='The omitted component still acts');ax[1].legend(frameon=False)
save(fig,'hidden-memory',{'t':t.tolist(),'q_positive':q.tolist(),'r_positive':np.exp(-2*t).tolist(),
 'status':'exact solution of two-component linear dynamics in nondimensional time'})

# 25: Exact endpoint distribution and seeded independent trajectories.
rng=np.random.default_rng(240925);steps=32;xs=np.arange(-32,33,2)
pmf=np.array([math.comb(32,k)/2**32 for k in range(33)])
draws=rng.choice([-1,1],size=(2048,32));paths=np.cumsum(draws,axis=1)
fig,ax=pair(3.6)
for j in range(5):ax[0].plot(np.arange(33),np.r_[0,paths[j]],alpha=.8,lw=1)
ax[0].set(xlabel='Step',ylabel='Position (lattice units)',title='Five independent sample paths')
for count,col,marker in [(32,ORANGE,'o'),(2048,BLUE,'s')]:
    freq=np.array([(paths[:count,-1]==x).mean() for x in xs])
    ax[1].plot(xs,freq,marker=marker,ls='none',ms=3,color=col,label=f'{count} sampled paths')
ax[1].plot(xs,pmf,color=INK,lw=1.5,label='Exact probability')
ax[1].set(xlabel='Endpoint (lattice units)',ylabel='Probability / observed fraction',title='The distribution is not one path');ax[1].legend(frameon=False,fontsize=8)
assert abs(pmf.sum()-1)<1e-12
save(fig,'paths-and-laws',{'seed':240925,'x':xs.tolist(),'exact_probability':pmf.tolist(),'endpoints':paths[:,-1].tolist(),
 'status':'exact binomial law plus seeded independent synthetic paths'})

# 29: Ideal balanced interferometer, real nonnegative path-record overlap.
phi=np.linspace(0,2*np.pi,361)
fig,ax=pair()
for eta,col,ls in [(1,BLUE,'-'),(.5,ORANGE,'--'),(0,GRAY,':')]:
    ax[0].plot(phi/np.pi,(1+eta*np.cos(phi))/2,color=col,ls=ls,label=f'Record overlap {eta:g}')
ax[0].set(xlabel='Relative phase / pi',ylabel='Probability at output 0',title='Phase matters when paths stay coherent');ax[0].legend(frameon=False,fontsize=8)
ax[1].plot(phi/np.pi,(1+np.cos(phi))/2,color=BLUE,label='Output 0')
ax[1].plot(phi/np.pi,(1-np.cos(phi))/2,color=ORANGE,ls='--',label='Output 1')
ax[1].axhline(1,color=GRAY,ls=':',label='Sum = 1')
ax[1].set(xlabel='Relative phase / pi',ylabel='Probability',title='The two outputs keep the total');ax[1].legend(frameon=False,fontsize=8)
save(fig,'amplitudes-records',{'phase':phi.tolist(),'coherent_p0':((1+np.cos(phi))/2).tolist(),
 'overlaps':[1,.5,0],'status':'ideal two-path quantum calculation, no measured counts'})

# 30: Three flat FLRW idealizations with equal a and H at reference time.
u=np.linspace(-.4,1,281)
models={'Pressureless matter':(1+1.5*u)**(2/3),'Radiation':np.sqrt(1+2*u),'Positive cosmological constant':np.exp(u)}
fig,ax=pair()
for (label,a),col,ls in zip(models.items(),[BLUE,ORANGE,GRAY],['-','--',':']):
    ax[0].plot(u,a,color=col,ls=ls,label=label)
    ax[1].plot(u,a-(1+u),color=col,ls=ls,label=label)
ax[0].scatter([0],[1],color=INK,zorder=4);ax[0].set(xlabel='Time offset × reference Hubble rate',ylabel='Relative scale factor',title='Same size and expansion rate now')
ax[0].legend(frameon=False,fontsize=7.6)
ax[1].axhline(0,color=GRAY,lw=.5);ax[1].set(xlabel='Time offset × reference Hubble rate',ylabel='Departure from shared tangent',title='Different surrounding histories')
save(fig,'cosmic-histories',{'u':u.tolist(),'scale_factors':{k:v.tolist() for k,v in models.items()},
 'status':'three flat single-component FLRW idealizations, not a fit to our universe'})

# 32: Exact miniature model-selection task, not a new network benchmark.
x=np.linspace(-1.1,1.1,221);train=np.array([-1.,0,1]);held=np.array([-.5,.5])
fig,ax=pair()
ax[0].plot(x,x*x,color=BLUE,label='Quadratic candidate');ax[0].axhline(2/3,color=ORANGE,ls='--',label='Best constant / linear fit')
ax[0].scatter(train,train**2,color=INK,label='Training points',zorder=4)
ax[0].scatter(held,held**2,facecolors='white',edgecolors=BLUE,marker='s',s=45,label='Declared test points',zorder=4)
ax[0].set(xlabel='Input x (dimensionless)',ylabel='Target or prediction',title='A declared, exact teaching task');ax[0].legend(frameon=False,fontsize=8)
err=[25/144,25/144,0]
ax[1].bar([1,2,3],err,color=[GRAY,ORANGE,BLUE]);ax[1].set_xticks([1,2,3],['Constant\n1 coefficient','Linear\n2 coefficients','Quadratic\n3 coefficients'])
ax[1].set(ylabel='Mean squared test error',title='Better fit costs another coefficient')
save(fig,'selection-menu',{'training_x':train.tolist(),'heldout_x':held.tolist(),'test_mse':err,
 'parameters':[1,2,3],'status':'exact noiseless polynomial example, not empirical architecture ranking'})

# 33: Same maintenance toy as earlier, now with explicit detection criterion.
t=np.linspace(0,65,651);M=np.where(t<=20,100,np.where(t<=35,100*np.exp(-.08*(t-20)),100+(100*np.exp(-1.2)-100)*np.exp(-.08*(t-35))))
obs=np.arange(0,66,5);Mo=np.interp(obs,t,M)
fig,ax=plt.subplots(2,1,figsize=(7.3,4.3),layout='constrained',sharex=True,gridspec_kw={'height_ratios':[2,1]})
ax[0].plot(t,M,color=BLUE);ax[0].axhline(50,color=ORANGE,ls='--',label='Declared detection threshold')
ax[0].scatter(obs,Mo,facecolors='white',edgecolors=INK,zorder=4,label='Samples every 5 time units')
ax[0].axvspan(20,35,color=GRAY,alpha=.1);ax[0].set(ylabel='Maintained amount (chosen unit)',title='Below detection is not necessarily destroyed');ax[0].legend(frameon=False,fontsize=8)
ax[1].plot(t,(M>=50).astype(float),color=ORANGE,drawstyle='steps-post',label='Continuous threshold rule')
ax[1].scatter(obs,(Mo>=50).astype(int),color=INK,label='Recorded detections')
ax[1].set(xlabel='Time (chosen unit)',ylabel='Detected?',yticks=[0,1],yticklabels=['No','Yes'])
save(fig,'continuity-detection',{'t':t.tolist(),'amount':M.tolist(),'sample_times':obs.tolist(),'sample_amount':Mo.tolist(),
 'threshold':50,'status':'analytic maintenance toy, no claim about biological identity'})

# 34: A reset breaks naive ordering by recorded clock values.
t=np.linspace(0,10,501);B=1.2*t+2-5*(t>=6)
ev=np.array([2.,5.,8.]);bv=1.2*ev+2-5*(ev>=6)
fig,ax=pair()
ax[0].plot(t,t,color=BLUE,label='Clock A');ax[0].plot(t[t<6],B[t<6],color=ORANGE,label='Clock B');ax[0].plot(t[t>=6],B[t>=6],color=ORANGE)
ax[0].plot([6,6],[9.2,4.2],color=ORANGE,ls=':');ax[0].set(xlabel='Reference time (s)',ylabel='Clock reading (s)',title='A clock can drift and reset');ax[0].legend(frameon=False)
for j,(e,b) in enumerate(zip(ev,bv)):
    ax[1].plot([e,b],[1,0],color=GRAY,lw=1)
    ax[1].scatter([e],[1],color=BLUE);ax[1].scatter([b],[0],color=ORANGE)
    ax[1].annotate(f'E{j+1}',(e,1),xytext=(0,7),ha='center',textcoords='offset points')
    ax[1].annotate(f'E{j+1}',(b,0),xytext=(0,-17),ha='center',textcoords='offset points')
ax[1].set(ylim=(-.45,1.45),yticks=[0,1],yticklabels=['Clock B','Clock A'],xlabel='Recorded value (s)',title='Sorting values changes the apparent order')
save(fig,'clock-relations',{'events_reference_s':ev.tolist(),'clock_B_s':bv.tolist(),'reset_s':6,
 'status':'specified clock toy, no claim of observed relativistic effect'})

# 35: Transcription of the actual E3 sealed report.
vals=[.0465176,.0222148,.0111033,.0169332]
fig,a=plt.subplots(figsize=(7.3,3.2),layout='constrained')
labels=['Shared-law\ncandidate','Constant\nbaseline','64-byte\nmemorizer','Disjoint-family\nbaseline']
bars=a.barh(np.arange(4),vals,color=[ORANGE,GRAY,BLUE,GRAY],height=.55)
a.set_yticks(np.arange(4),labels);a.invert_yaxis();a.set(xlim=(0,.059),xlabel='Reported mean squared error (lower is better)',title='This candidate did not transfer successfully')
for bar,val in zip(bars,vals):a.text(val+.001,bar.get_y()+bar.get_height()/2,f'{val:.7f}',va='center',fontsize=9)
save(fig,'shared-law-null',{'reported_mse':dict(zip(labels,vals)),
 'source':'XFAM_E3_SEALED_RESULT_REPORT_2026_08_26.md','status':'transcribed reported results, no rerun','uncertainty':'no error bars inferred from scalar report'})

# 36: A set alone does not assign its own probability.
k=np.linspace(0,1,301);fig,ax=pair()
for a,density,title,p in [(ax[0],np.ones_like(k),'Uniform weighting',.5),(ax[1],2*k,'Weight increasing with k',.75)]:
    a.plot(k,density,color=BLUE);a.fill_between(k,density,where=k>=.5,color=ORANGE,alpha=.3)
    a.axvline(.5,color=GRAY,ls=':');a.set(xlabel='Toy rule parameter k',ylabel='Probability density',title=f'{title}: supported mass {p:g}')
save(fig,'world-measures',{'k':k.tolist(),'uniform_density':np.ones_like(k).tolist(),'rising_density':(2*k).tolist(),
 'support_condition':'k >= 0.5','masses':[.5,.75],'status':'mathematical weighting example, not a measure on actual universes'})
data_path=OUT.parent/'figure-data.json'
data_path.write_text(json.dumps(data,indent=2),encoding='utf-8')
(OUT.parent/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
(OUT.parent/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.pdf')},indent=2),encoding='utf-8')
print(f'{len(data)} figures generated. Analytic checks: {checks}')
