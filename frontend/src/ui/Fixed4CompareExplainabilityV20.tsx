
import React from 'react'
import Plot from 'react-plotly.js'

type Obj = Record<string, any>
const DEFAULT_BATTERIES = ['B0018','B0033','B0042','B0043','B0055']
const METRICS: [string,string][] = [
  ['soh','SoH (%)'],
  ['capacity_pct','Capacity (% of initial)'],
  ['impedance_sum','Impedance sum (Ω)'],
  ['dcr','DCR (Ω)'],
  ['thermal_stress','Thermal stress'],
  ['temperature_mean','Temp mean (°C)'],
  ['lli','LLI'],
  ['lam','LAM'],
]
function num(v:any){ const n=Number(v); return Number.isFinite(n)?n:NaN }
function nearest(xs:number[], x:number){ if(!xs.length) return -1; let b=0; for(let i=1;i<xs.length;i++) if(Math.abs(xs[i]-x)<Math.abs(xs[b]-x)) b=i; return b }
async function json(url:string){ const r=await fetch(url); const t=await r.text(); if(!r.ok) throw new Error(`${r.status} ${r.statusText}: ${t.slice(0,220)}`); return JSON.parse(t) }
function unwrapPre(raw:any){ const o=raw?.payload??raw?.item??raw?.data??raw?.precomputed??raw; return { cycles:(o?.query?.cycle??o?.cycles??[]).map(num), pred:(o?.pred?.mean??o?.rul_pred??o?.predRUL??[]).map(num), truth:(o?.query?.true_rul??o?.rul_true??o?.trueRUL??[]).map(num), std:(o?.pred?.std??o?.rul_std??o?.std??[]).map(num) } }

export function Fixed4CompareTabV20({battery,cycle,batteries}:{battery:string,cycle:number,batteries?:string[]}){
  const opts=React.useMemo(()=>Array.from(new Set([battery,...(batteries||[]),...DEFAULT_BATTERIES].filter(Boolean))),[battery,batteries])
  const [sel,setSel]=React.useState<string[]>(()=>Array.from(new Set([battery||'B0018','B0043'])).slice(0,4))
  const [metric,setMetric]=React.useState('soh')
  const [cohort,setCohort]=React.useState('all')
  const [band,setBand]=React.useState(true)
  const [data,setData]=React.useState<any>(null)
  const [err,setErr]=React.useState<string|null>(null)
  const [load,setLoad]=React.useState(false)
  React.useEffect(()=>{ if(battery) setSel(p=>p.includes(battery)?p:[battery,...p].slice(0,4)) },[battery])
  React.useEffect(()=>{ let ok=true; (async()=>{ setLoad(true); setErr(null); try{ const u=`/api/fixed4/compare-v20?batteries=${encodeURIComponent(sel.join(','))}&metric=${metric}&cohort=${cohort}&cycle=${cycle||''}`; const d=await json(u); if(ok) setData(d) }catch(e:any){ if(ok) setErr(String(e?.message||e)) } finally{ if(ok) setLoad(false) } })(); return()=>{ok=false} },[sel.join(', '), metric, cohort])
  const traces:any[]=[]
  if(band&&data?.band?.x?.length){ traces.push({x:data.band.x,y:data.band.q75,type:'scatter',mode:'lines',line:{width:0},showlegend:false,hoverinfo:'skip',name:'q75'}); traces.push({x:data.band.x,y:data.band.q25,type:'scatter',mode:'lines',fill:'tonexty',line:{width:0},opacity:.25,hoverinfo:'skip',name:'IQR (cohort)'}); traces.push({x:data.band.x,y:data.band.median,type:'scatter',mode:'lines',name:'Median (cohort)',line:{dash:'dash'}}) }
  ;(data?.series||[]).forEach((s:any)=>traces.push({x:s.x||[],y:s.y||[],type:'scatter',mode:'lines',name:s.battery,line:{width:String(s.battery)===String(battery)?3:2},hovertemplate:`${s.battery}<br>Cycle: %{x}<br>${data?.metric_label||metric}: %{y:.4f}<extra></extra>`}))
  const rows=(data?.series||[]).map((s:any)=>{ const xs=(s.x||[]).map(num), ys=(s.y||[]).map(num), i=nearest(xs,Number(cycle)); return {b:s.battery,c:i>=0?xs[i]:NaN,v:i>=0?ys[i]:NaN,m:s.missing} })
  return <div style={{overflowY:'auto',paddingRight:4}}>
    <h3>🆚 Compare batteries (Geotab-style)</h3>
    <p style={{fontSize:12,color:'#666'}}>Fixed4-compatible: metric selector, battery lines, cohort median/IQR expected band, current-cycle red dashed line.</p>
    <div style={{display:'flex',gap:8,alignItems:'center',flexWrap:'wrap',padding:10,background:'#f8fafc',border:'1px solid #e5e7eb',borderRadius:8,marginBottom:12}}>
      <b style={{fontSize:12}}>비교할 배터리</b>{opts.map(b=><label key={b} style={{fontSize:12,padding:'5px 8px',borderRadius:999,border:sel.includes(b)?'1px solid #1976d2':'1px solid #ddd',background:sel.includes(b)?'#1976d2':'white',color:sel.includes(b)?'white':'#333'}}><input type="checkbox" checked={sel.includes(b)} onChange={e=>e.target.checked?setSel(p=>Array.from(new Set([...p,b])).slice(0,4)):setSel(p=>p.filter(x=>x!==b))}/> {b}</label>)}
      <select value={metric} onChange={e=>setMetric(e.target.value)}>{METRICS.map(([k,l])=><option key={k} value={k}>{l}</option>)}</select>
      <select value={cohort} onChange={e=>setCohort(e.target.value)}><option value="all">전체(All)</option><option value="temp_le_10">주변온도 ≤ 10°C</option><option value="temp_10_25">10–25°C</option><option value="temp_ge_25">≥ 25°C</option></select>
      <label style={{fontSize:12}}><input type="checkbox" checked={band} onChange={e=>setBand(e.target.checked)}/> 기대 범위</label>
    </div>
    {load&&<div>Loading fixed4 compare data...</div>}{err&&<div style={{color:'#d32f2f'}}>Compare error: {err}</div>}
    <table style={{width:'100%',borderCollapse:'collapse',fontSize:11,marginBottom:12}}><thead><tr style={{background:'#eee'}}><th>Battery</th><th>Nearest cycle</th><th>{data?.metric_label||metric}</th><th>Line</th></tr></thead><tbody>{rows.map((r:any)=><tr key={r.b}><td><b>{r.b}</b></td><td style={{textAlign:'center'}}>{Number.isFinite(r.c)?r.c:'N/A'}</td><td style={{textAlign:'center'}}>{Number.isFinite(r.v)?r.v.toFixed(4):(r.m?'missing column':'N/A')}</td><td style={{textAlign:'center'}}>{String(r.b)===String(battery)?'selected · width 3':'width 2'}</td></tr>)}</tbody></table>
    {traces.length?<Plot data={traces} layout={{height:420,margin:{l:55,r:20,t:20,b:45},xaxis:{title:'Cycle'},yaxis:{title:data?.metric_label||metric},hovermode:'x unified',legend:{orientation:'h',y:1.14,x:0},shapes:Number.isFinite(Number(cycle))?[{type:'line',xref:'x',yref:'paper',x0:Number(cycle),x1:Number(cycle),y0:0,y1:1,line:{color:'red',dash:'dash',width:2}}]:[]}} config={{responsive:true,displayModeBar:false}} useResizeHandler style={{width:'100%'}}/>:<div style={{padding:24,textAlign:'center',color:'#999'}}>No plot data.</div>}
  </div>
}

export function Fixed4ExplainabilityTabV20({battery,rRatio,cycle}:{battery:string,rRatio:string|number,cycle:number}){
  const [pre,setPre]=React.useState<any>(null),[deg,setDeg]=React.useState<any>(null),[shap,setShap]=React.useState<any>(null)
  const [err,setErr]=React.useState<string|null>(null),[load,setLoad]=React.useState(false),[inf,setInf]=React.useState(false),[log,setLog]=React.useState<any>(null),[topK,setTopK]=React.useState(12)
  async function loadData(){ if(!battery) return; setLoad(true); setErr(null); try{ const [p,d,s]=await Promise.all([json(`/api/battery/${battery}/precomputed?r_ratio=${encodeURIComponent(String(rRatio))}`).catch(e=>({__error:String(e?.message||e)})),json(`/api/battery/${battery}/degradation-monitoring?r_ratio=${encodeURIComponent(String(rRatio))}`).catch(e=>({__error:String(e?.message||e)})),json('/api/fixed4/shap-v21').catch(()=>null)]); setPre(p?.__error?null:unwrapPre(p)); setDeg(d?.__error?null:d); setShap(s); if(p?.__error) setErr('precomputed: '+p.__error) } finally{ setLoad(false) } }
  React.useEffect(()=>{loadData()},[battery,rRatio])
  const i=pre?.cycles?nearest(pre.cycles,Number(cycle)):-1, pred=i>=0?num(pre.pred?.[i]):NaN, truth=i>=0?num(pre.truth?.[i]):NaN, sig=i>=0?num(pre.std?.[i]):NaN
  const res=Number.isFinite(pred)&&Number.isFinite(truth)?pred-truth:NaN, u2=Number.isFinite(sig)?2*sig:NaN, conf=Number.isFinite(u2)&&Math.abs(pred)>1?Math.max(0,Math.min(100,100*(1-Math.min(1,u2/Math.abs(pred))))):NaN
  const cap=num(deg?.cap_min_z), dcr=num(deg?.dcr_max_z)
  const card=(t:string,v:string,s='',c='#111')=><div style={{padding:12,background:'white',border:'1px solid #e5e7eb',borderRadius:8}}><div style={{fontSize:11,color:'#666',fontWeight:800}}>{t}</div><div style={{fontSize:22,fontWeight:900,color:c}}>{v} <span style={{fontSize:12,opacity:.7}}>{s}</span></div></div>
  async function runLive(){ setInf(true); setLog(null); const st=Date.now(); try{ const out=await json(`/api/live-reinfer-v20/${battery}?r_ratio=${encodeURIComponent(String(rRatio))}&timeout=360`); setLog({elapsed_sec_client:((Date.now()-st)/1000).toFixed(1),...out}); await loadData() }catch(e:any){ setLog({ok:false,error:String(e?.message||e)}) }finally{ setInf(false)} }
  const shapItems=(shap?.items||[]).slice(0,Math.max(5,Math.min(30,topK))).slice().reverse()
  return <div style={{overflowY:'auto',paddingRight:4}}>
    <h3>🧠 Explainability</h3><p style={{fontSize:12,color:'#666'}}>Prediction confidence + model architecture + fixed4 global SHAP importance.</p>
    {load&&<div>Loading explainability data...</div>}{err&&<div style={{color:'#b45309',background:'#fffbeb',padding:8}}>Data warning: {err}</div>}
    <section style={{marginBottom:16,padding:12,background:'#f8fafc',border:'1px solid #e5e7eb',borderRadius:8}}><h4>Prediction Confidence</h4><div style={{display:'grid',gridTemplateColumns:'repeat(4,minmax(0,1fr))',gap:8}}>{card('Predicted RUL',Number.isFinite(pred)?pred.toFixed(1):'N/A','cycles','#1976d2')}{card('True RUL',Number.isFinite(truth)?truth.toFixed(1):'N/A','cycles')}{card('Uncertainty',Number.isFinite(u2)?`±${u2.toFixed(1)}`:'N/A','2σ','#ff6f00')}{card('Confidence',Number.isFinite(conf)?`${conf.toFixed(0)}%`:'N/A',Number.isFinite(conf)?(conf>=80?'High':conf>=55?'Medium':'Low'):'')}</div><div style={{marginTop:10,fontSize:12}}><b>Residual:</b> {Number.isFinite(res)?`${res>=0?'+':''}${res.toFixed(1)} cycles`:'Observed RUL unavailable at selected cycle.'}</div></section>
    <div style={{display:'grid',gridTemplateColumns:'1.15fr .85fr',gap:12,marginBottom:16}}><section style={{padding:12,border:'1px solid #e5e7eb',borderRadius:8}}><h4>Model Architecture</h4><p><b>Backbone:</b> CEEMDAN–Transformer–DNN. CEEMDAN decomposes noisy capacity trajectories into local regeneration components and a global residual trend.</p><p><b>Meta-learning:</b> BMAML-SVGD is applied on top for few-shot adaptation and uncertainty-aware RUL prediction.</p></section><section style={{padding:12,border:'1px solid #e5e7eb',borderRadius:8}}><h4>Initialize & Re-inference</h4><p style={{fontSize:12}}>Runs live backend runner. No silent precomputed fallback.</p><button onClick={runLive} disabled={inf||!battery} style={{padding:'9px 12px',background:inf?'#bbb':'#1976d2',color:'white',border:0,borderRadius:6,width:'100%',fontWeight:900}}>{inf?'Running live BMAML-SVGD inference...':'Initialize & Re-inference'}</button>{log&&<pre style={{maxHeight:150,overflow:'auto',background:'#0f172a',color:log.ok===false?'#fecaca':'#d1fae5',padding:8,borderRadius:6,fontSize:10}}>{JSON.stringify(log,null,2)}</pre>}</section></div>
    <section style={{marginBottom:16,padding:12,border:'1px solid #e5e7eb',borderRadius:8}}><h4>Global feature importance</h4>{shapItems.length?<><label>Top-K <input type="range" min={5} max={Math.min(30,shap?.items?.length||30)} value={topK} onChange={e=>setTopK(Number(e.target.value))}/><b>{topK}</b></label><Plot data={[{type:'bar',orientation:'h',x:shapItems.map((x:any)=>x.importance),y:shapItems.map((x:any)=>x.feature),name:'global importance'}]} layout={{height:380,margin:{l:120,r:20,t:20,b:35},xaxis:{title:'Importance'},yaxis:{title:'Feature'}}} config={{responsive:true,displayModeBar:false}} useResizeHandler style={{width:'100%'}}/><div style={{fontSize:10,color:'#777'}}>※ 이 중요도는 특정 cycle의 anomaly 원인이 아니라, BMAML/sequence model 전체 입력 feature에 대한 전역(global) 중요도입니다.</div></>:<div style={{color:'#777'}}>SHAP 전역 중요도 파일을 찾지 못했습니다. (shap_outputs/*.json)</div>}</section>
    <section style={{marginBottom:16,padding:12,background:'#fff7ed',border:'1px solid #fed7aa',borderRadius:8}}><h4>Degradation evidence bridge</h4><div style={{display:'grid',gridTemplateColumns:'repeat(3,minmax(0,1fr))',gap:8}}>{card('Degradation status',String(deg?.status||'N/A'),'',deg?.status==='major-anomaly'?'#dc2626':'#16a34a')}{card('Capacity min z',Number.isFinite(cap)?cap.toFixed(2):'N/A','prefix',Number.isFinite(cap)&&cap<=-3.5?'#dc2626':'#111')}{card('DCR max z',Number.isFinite(dcr)?dcr.toFixed(2):'N/A','prefix',Number.isFinite(dcr)&&dcr>=4?'#dc2626':'#111')}</div></section>
    <section style={{marginBottom:16,padding:12,background:'#f8fafc',border:'1px solid #e5e7eb',borderRadius:8}}><h4>Methods & References</h4><div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:8,fontSize:11,lineHeight:1.5}}><div><b>CEEMDAN–Transformer–DNN</b><br/>Early lithium-ion battery RUL prediction using decomposition, Transformer modules, and DNN residual trend modeling.</div><div><b>BMAML-SVGD / uncertainty</b><br/>Few-shot RUL prediction using Bayesian meta-learning, particle uncertainty, and SVGD-style adaptation.</div><div><b>Resistance diagnostics</b><br/>Early resistance signals can be informative diagnostic features for lifetime prediction.</div><div><b>EV battery aging data</b><br/>RPT, HPPC, EIS, and capacity diagnostics motivate multi-feature monitoring.</div></div></section>
  </div>
}
