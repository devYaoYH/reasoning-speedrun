/**
 * Compare recorded time to 18 correct across canonical attempts. Use /results
 * through src.viewer_server after syncing evidence and annotating metadata.json.
 * Plot attempts by start timestamp with a dotted best-so-far frontier, show the
 * serial grader floor, and expose changed
 * controls without treating single-run differences as isolated causal effects.
 */
const $ = selector => document.querySelector(selector);
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const sec = v => v == null ? '—' : `${v.toFixed(2)}s`;
// Model identity includes weight quantization and activation dtype, not run order.
const modelStyles = new Map([
  ['WeiboAI/VibeThinker-3B|none|bfloat16',{label:'VibeThinker 3B · BF16',color:'#147d65'}],
  ['r0b0tlab/VibeThinker-3B-NVFP4|modelopt_fp4|bfloat16',{label:'VibeThinker 3B · NVFP4',color:'#b37b30'}],
  ['Qwen/Qwen3.5-4B|none|bfloat16',{label:'Qwen 3.5 4B · BF16',color:'#54729b'}],
  ['Qwen/Qwen3.5-35B-A3B-GPTQ-Int4|gptq int4|bfloat16',{label:'Qwen 3.5 35B-A3B · GPTQ Int4',color:'#865b89'}]
]);
function modelStyle(row) {
  const model=row.metadata?.model??{},id=model.id??'Unknown model';
  const quant=(model.quantization??'unknown').toLowerCase(),dtype=(model.activation_dtype??'unknown').toLowerCase();
  const key=`${id}|${quant}|${dtype}`,known=modelStyles.get(key);
  if(known)return {key,...known};
  let hash=0;for(const char of key)hash=(Math.imul(hash,31)+char.charCodeAt(0))>>>0;
  return {key,label:`${id.split('/').at(-1)} · ${quant==='none'?dtype:quant}`,color:`hsl(${hash%360} 45% 38%)`};
}
function modelLegend(rows) {
  const variants=new Map(rows.map(r=>{const style=modelStyle(r);return [style.key,style];}));
  return [...variants.values()].sort((a,b)=>a.label.localeCompare(b.label))
    .map(s=>`<span class="model-key"><i style="background:${s.color}"></i>${esc(s.label)}</span>`).join('');
}
const attemptLink = row => `/?attempt=${encodeURIComponent(row.id)}`;
// Historical points 2 and 3 are separate baselines; identify them by stable IDs.
const separateBaselines = new Set(['20261003T202152.418590Z','20261003T203338.063138Z']);
let results;
function benchmarkRows(rows, year) {
  return rows.filter(r => (Object.hasOwn(r, 'benchmark_year') ? r.benchmark_year :
    r.metadata?.provenance?.dataset?.year ?? 2025) === Number(year));
}
// The page selects a dataset (AIME year or a custom dataset id); benchmarkRows stays the AIME-year filter.
function benchmarkKey(r) {
  const dataset = r.metadata?.provenance?.dataset;
  // An explicit null year means a generic dataset; only a missing year defaults to AIME 2025.
  const year = Object.hasOwn(r, 'benchmark_year') ? r.benchmark_year : (dataset && Object.hasOwn(dataset, 'year') ? dataset.year : 2025);
  return year == null ? `dataset:${r.benchmark_id ?? r.metadata?.provenance?.dataset?.id ?? 'unknown'}` : `aime:${year}`;
}
const keyOfCluster = c => c.benchmark_year == null ? `dataset:${c.benchmark_id ?? 'unknown'}` : `aime:${c.benchmark_year}`;
function benchmarkRowsByKey(rows, key) { return rows.filter(r => benchmarkKey(r) === key); }
function benchmarkLabel(key, rows) {
  const matching = rows.filter(r => benchmarkKey(r) === key);
  const first = matching.find(r => r.benchmark_role) ?? matching[0] ?? {};
  return key.startsWith('aime:') ? `AIME ${key.slice(5)}${first.benchmark_role ? ' · ' + first.benchmark_role : ''}` : key.slice(8);
}
function populateBenchmarkFilter() {
  const select = $('#benchmark-year'), current = select.value;
  const rows = results.attempts, keys = [...new Set(rows.map(benchmarkKey))];
  const newest = [...rows].sort((a, b) => Date.parse(b.attempt_started_at_utc ?? b.started_at_utc ?? 0) - Date.parse(a.attempt_started_at_utc ?? a.started_at_utc ?? 0))[0];
  select.innerHTML = keys.map(k => `<option value="${esc(k)}">${esc(benchmarkLabel(k, rows))}</option>`).join('');
  select.value = keys.includes(current) ? current : (newest ? benchmarkKey(newest) : (keys[0] ?? ''));
}
function clusterRows(rows, id) { return id==='all'?rows:rows.filter(r=>r.cluster?.family_id===id); }
function selectedRows() { return clusterRows(benchmarkRowsByKey(results.attempts, $('#benchmark-year').value), $('#attempt-cluster').value); }
function attemptHistory(rows, numberingRows=rows) {
  const measured=numberingRows.filter(r=>r.time_to_18_s!=null && Number.isFinite(r.time_to_18_s) && r.time_to_18_s>=0)
    .map(r=>({...r,start_ms:Date.parse(r.attempt_started_at_utc??r.started_at_utc)}))
    .filter(r=>Number.isFinite(r.start_ms)).sort((a,b)=>a.start_ms-b.start_ms || a.id.localeCompare(b.id))
    .map((r,i)=>({...r,plot_number:i+1}));
  const visible=new Set(rows.map(r=>r.id));
  const points=measured.filter(r=>visible.has(r.id) && !separateBaselines.has(r.id));
  let best=Infinity;
  const frontier=points.filter(r=>{if(r.time_to_18_s<best){best=r.time_to_18_s;return true;}return false;});
  return {points,frontier,baselines:measured.filter(r=>visible.has(r.id) && separateBaselines.has(r.id))};
}
const historyDate=new Intl.DateTimeFormat('en-US',{timeZone:'America/Los_Angeles',month:'short',day:'numeric',hour:'numeric',minute:'2-digit',second:'2-digit',timeZoneName:'short'});
const historyTick=new Intl.DateTimeFormat('en-US',{timeZone:'America/Los_Angeles',hour:'numeric',minute:'2-digit'});
function comparisonPlot(history) {
  const {points,frontier}=history;
  if (!points.length) return '<div class="chart-empty">No attempt has both a measured time to 18 and a recorded start timestamp.</div>';
  const width=1080,height=410,left=86,right=30,top=40,bottom=80;
  const span=Math.max(points.at(-1).start_ms-points[0].start_ms,60000),padding=span*.035;
  const xmin=points[0].start_ms-padding,xmax=points.at(-1).start_ms+padding;
  const min=50,peak=Math.max(60,...points.map(r=>r.time_to_18_s));
  const step=Math.max(10,Math.ceil((peak-min)/70)*10),max=min+Math.ceil((peak-min)/step)*step;
  const x=t=>left+(t-xmin)/(xmax-xmin)*(width-left-right);
  const y=t=>height-bottom-(t-min)/(max-min)*(height-top-bottom);
  let svg=`<svg class="comparison-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Attempt start timestamp versus end-to-end time to 18 verified correct answers, with a dotted best-so-far Pareto step and a horizontal 54 second grader floor"><text x="${left}" y="18">Time to 18 verified correct answers (seconds)</text>`;
  for(let t=min;t<=max;t+=step){
    svg+=`<line class="grid" x1="${left}" x2="${width-right}" y1="${y(t)}" y2="${y(t)}"/><text text-anchor="end" x="${left-12}" y="${y(t)+4}">${t}s</text>`;
  }
  for(let i=0;i<=6;i++){
    const stamp=xmin+(xmax-xmin)*i/6;
    svg+=`<line class="grid" x1="${x(stamp)}" x2="${x(stamp)}" y1="${top}" y2="${height-bottom}"/><text text-anchor="middle" x="${x(stamp)}" y="${height-bottom+26}">${esc(historyTick.format(stamp))}</text>`;
  }
  svg+=`<text text-anchor="middle" x="${left+(width-left-right)/2}" y="${height-12}">Attempt started · America/Los_Angeles · ${esc(new Intl.DateTimeFormat('en-US',{timeZone:'America/Los_Angeles',month:'short',day:'numeric',year:'numeric'}).format(points[0].start_ms))}</text>`;
  const floor=results.reference_floor_s;
  svg+=`<line class="floor-line" x1="${left}" x2="${width-right}" y1="${y(floor)}" y2="${y(floor)}"/><text class="floor-label" x="${left+8}" y="${y(floor)+17}">54s grader floor · 18 × 3s</text>`;
  let d=`M ${x(frontier[0].start_ms)} ${y(frontier[0].time_to_18_s)}`;
  frontier.slice(1).forEach(r=>{d+=` H ${x(r.start_ms)} V ${y(r.time_to_18_s)}`;});
  d+=` H ${x(xmax)}`;
  svg+=`<path class="pareto-frontier" d="${d}"/>`;
  points.forEach(r=>{
    const description=`${r.metadata.label} · ${sec(r.time_to_18_s)} · started ${historyDate.format(r.start_ms)} · ${r.metadata.intervention.label}`;
    svg+=`<a class="attempt-dot-link" href="${attemptLink(r)}" aria-label="${esc(description)}"><title>${esc(description)}</title><circle class="attempt-dot" data-attempt="${esc(r.id)}" cx="${x(r.start_ms)}" cy="${y(r.time_to_18_s)}" r="11" fill="${modelStyle(r).color}"/><text class="dot-number" text-anchor="middle" x="${x(r.start_ms)}" y="${y(r.time_to_18_s)+4}">${r.plot_number}</text></a>`;
  });
  return svg+'</svg>';
}
function historyLegend(history) {
  const bestIds=new Set(history.frontier.map(r=>r.id));
  return history.points.map(r=>`<a class="history-item" href="${attemptLink(r)}"><span class="history-number" style="background:${modelStyle(r).color}">${r.plot_number}</span><span><strong>${esc(r.metadata.label)}</strong><small>${sec(r.time_to_18_s)} · ${esc(historyDate.format(r.start_ms))}${bestIds.has(r.id)?' · new best':''}</small><small>${esc(r.metadata.intervention.label)}</small></span></a>`).join('');
}
function initialLatencyHistory(rows, numberingRows=rows) {
  const inventory=numberingRows.map(r=>({...r,start_ms:Date.parse(r.attempt_started_at_utc??r.started_at_utc)}))
    .filter(r=>r.metadata && Number.isFinite(r.start_ms))
    .sort((a,b)=>a.start_ms-b.start_ms || a.id.localeCompare(b.id))
    .map((r,i)=>({...r,initial_number:i+1}));
  const visible=new Set(rows.map(r=>r.id));
  const points=inventory.filter(r=>visible.has(r.id) && Number.isFinite(r.first_grader_request_s) && r.first_grader_request_s>=0);
  return {points,unavailable:rows.filter(r=>!points.some(p=>p.id===r.id))};
}
function initialLatencyPlot(history) {
  const {points}=history;
  if(!points.length)return '<div class="chart-empty">No recorded first grader request timing is available.</div>';
  const width=1080,height=410,left=86,right=30,top=40,bottom=80;
  const span=Math.max(points.at(-1).start_ms-points[0].start_ms,60000),padding=span*.035;
  const xmin=points[0].start_ms-padding,xmax=points.at(-1).start_ms+padding;
  const peak=Math.max(5,...points.map(r=>r.first_grader_request_s));
  const step=Math.max(5,Math.ceil(peak/30)*5),max=Math.ceil(peak/step)*step;
  const x=t=>left+(t-xmin)/(xmax-xmin)*(width-left-right);
  const y=t=>height-bottom-t/max*(height-top-bottom);
  let svg=`<svg class="comparison-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Attempt start timestamp versus time to first grader request, from the official clock start after warmup"><text x="${left}" y="18">Time to first grader request (seconds)</text>`;
  for(let t=0;t<=max;t+=step){
    svg+=`<line class="grid" x1="${left}" x2="${width-right}" y1="${y(t)}" y2="${y(t)}"/><text text-anchor="end" x="${left-12}" y="${y(t)+4}">${t}s</text>`;
  }
  for(let i=0;i<=6;i++){
    const stamp=xmin+(xmax-xmin)*i/6;
    svg+=`<line class="grid" x1="${x(stamp)}" x2="${x(stamp)}" y1="${top}" y2="${height-bottom}"/><text text-anchor="middle" x="${x(stamp)}" y="${height-bottom+26}">${esc(historyTick.format(stamp))}</text>`;
  }
  svg+=`<text text-anchor="middle" x="${left+(width-left-right)/2}" y="${height-12}">Attempt started · America/Los_Angeles · ${esc(new Intl.DateTimeFormat('en-US',{timeZone:'America/Los_Angeles',month:'short',day:'numeric',year:'numeric'}).format(points[0].start_ms))}</text>`;
  points.forEach(r=>{
    const description=`${r.metadata.label} · first grader request ${sec(r.first_grader_request_s)} · started ${historyDate.format(r.start_ms)} · ${r.first_grader_request?.source??'recorded request timestamp'}`;
    svg+=`<a class="attempt-dot-link" href="${attemptLink(r)}" aria-label="${esc(description)}"><title>${esc(description)}</title><circle class="attempt-dot initial-dot" data-attempt="${esc(r.id)}" cx="${x(r.start_ms)}" cy="${y(r.first_grader_request_s)}" r="10" fill="${modelStyle(r).color}"/><text class="dot-number initial-number" text-anchor="middle" x="${x(r.start_ms)}" y="${y(r.first_grader_request_s)+3}">A${r.initial_number}</text></a>`;
  });
  return svg+'</svg>';
}
function initialLatencyLegend(history) {
  return history.points.map(r=>`<a class="history-item" href="${attemptLink(r)}"><span class="history-number initial-number" style="background:${modelStyle(r).color}">A${r.initial_number}</span><span><strong>${esc(r.metadata.label)}</strong><small>${sec(r.first_grader_request_s)} · ${esc(historyDate.format(r.start_ms))}</small><small>${esc(r.metadata.intervention.label)} · ${esc(r.status)}</small></span></a>`).join('');
}
function interventionCards(rows) {
  const comparisons=rows.filter(r=>r.metadata && r.comparison?.saved_s!=null);
  if(!comparisons.length)return '<p class="detail-note">Annotate a reference attempt in metadata.json to compare interventions.</p>';
  return comparisons.map(r=>{const m=r.metadata,c=r.comparison,faster=c.saved_s>=0;return `<article class="intervention-card"><h3>${esc(m.intervention.label)}</h3><small>${esc(m.label)} vs ${esc(c.reference_label)}</small><div class="delta ${faster?'':'slower'}">${sec(Math.abs(c.saved_s))} ${faster?'faster':'slower'}</div><small>${Math.abs(c.reduction_pct).toFixed(1)}% ${faster?'reduction':'increase'} in observed time to 18</small><ul>${m.intervention.changed_variables.map(v=>`<li>${esc(v)}</li>`).join('')}</ul><p>${esc(m.intervention.comparison_note)}</p><a href="${attemptLink(r)}">Inspect attempt ↗</a></article>`;}).join('');
}
function renderProgress(){
  const rows=selectedRows().filter(r=>r.events.length && r.metadata);
  if(!rows.length){$('#progress-model-legend').innerHTML='';$('#curve-legend').innerHTML='';$('#progress-plot').innerHTML='<div class="chart-empty">No timestamped positive verdicts available.</div>';return;}
  const window=$('#curve-window').value, max=window==='all'?Math.max(54,...rows.flatMap(r=>r.events.map(e=>e.elapsed_s))):Number(window);
  const w=1080,h=315,left=55,right=25,top=35,bottom=40;
  const x=t=>left+t/max*(w-left-right),y=n=>h-bottom-n/18*(h-top-bottom);
  let svg=`<svg class="progress-chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="Distinct correct questions over official elapsed seconds"><text x="${left}" y="18">Verified correct questions</text>`;
  for(let i=0;i<=6;i++){svg+=`<line class="grid" x1="${left}" x2="${w-right}" y1="${y(i*3)}" y2="${y(i*3)}"/><text x="${left-12}" y="${y(i*3)+4}" text-anchor="end">${i*3}</text><text text-anchor="middle" x="${x(max*i/6)}" y="${h-10}">${Math.round(max*i/6)}s</text>`;}
  svg+=`<line class="floor-line" x1="${x(54)}" x2="${x(54)}" y1="${top}" y2="${h-bottom}"/><text class="floor-label" x="${x(54)+7}" y="${top+16}">54s floor</text>`;
  rows.forEach(r=>{let d=`M ${x(0)} ${y(0)}`;r.events.slice(0,18).forEach((e,n)=>{if(e.elapsed_s<=max)d+=` H ${x(e.elapsed_s)} V ${y(n+1)}`;});d+=` H ${x(Math.min(max,r.settlement_s??r.events.at(-1).elapsed_s))}`;svg+=`<path d="${d}" fill="none" stroke="${modelStyle(r).color}" stroke-width="2.5"><title>${esc(r.metadata.label)}</title></path>`;});
  $('#progress-plot').innerHTML=svg+'</svg>';$('#progress-model-legend').innerHTML=modelLegend(rows);
  $('#curve-legend').innerHTML=rows.map(r=>`<span><i style="background:${modelStyle(r).color}"></i>${esc(r.metadata.label)}${r.time_to_18_s==null?' · target unmet':''}</span>`).join('');
}
function controlsTable(rows){
  return rows.map(r=>{const m=r.metadata;if(!m)return `<tr><td><a href="${attemptLink(r)}">${esc(r.id)}</a></td><td colspan="6">Invalid evidence; see warning above.</td></tr>`;
    const hp=m.controls.hyperparameters,g=m.gpu,c=r.comparison;
    const envelope=g.memory_utilization==null?'Unrecorded':`${Math.round(g.memory_utilization*100)}%${g.configured_envelope_mib!=null?` · ${(g.configured_envelope_mib/1024).toFixed(0)} GiB`:''}`;
    const changed=c?.changed_controls.map(v=>`${v.variable}: ${JSON.stringify(v.before)} → ${JSON.stringify(v.after)}`).join('\n');
    return `<tr><td><a href="${attemptLink(r)}">${esc(m.label)}</a><small>${esc(r.id)}<br>${esc(m.controls.dataset)} · ${esc(r.benchmark_role??'role unrecorded')}<br>${esc(m.model.id)}<br>${esc(m.model.quantization??'Quantization unrecorded')} · ${esc(m.model.activation_dtype??'dtype unrecorded')}</small></td><td>${esc(m.intervention.label)}<small>${esc(r.attempt_status)}</small></td><td><span class="result-time">${sec(r.time_to_18_s)}</span><small class="${r.time_to_18_s==null?'unmet':''}">${esc(r.status)}</small><small>Settlement: ${sec(r.settlement_s)}</small></td><td>${r.solved??'—'} / ${m.controls.question_indices.length||'—'}</td><td>${hp.parallelism??'—'} × ${hp.rollouts??'—'}<small>${hp.first_pass_max_tokens??'—'} first-pass tokens<br>${hp.max_attempts_per_question??'—'} requests / question</small></td><td>${envelope}<small>${esc(g.device??'Device unrecorded')}</small></td><td><details><summary>Inspect controls</summary><p>${esc(m.intervention.comparison_note)}</p>${r.error?`<p class="unmet">${esc(r.error)}</p>`:''}<small>Runner: ${esc(m.runner.module)}<br>${esc(m.runner.version)}<br>Source: ${esc(m.runner.git_commit)}<br>Context: ${m.controls.max_context_tokens??'—'} tokens<br>Seed: ${hp.seed??'—'} · T ${hp.temperature??'—'} · top-p ${hp.top_p??'—'}</small>${changed?`<h4>Changed recorded controls</h4><pre>${esc(changed)}</pre><h4>Matched recorded controls</h4><pre>${esc(c.matched_controls.join('\n'))}</pre>`:''}<h4>Full metadata</h4><pre>${esc(JSON.stringify(m,null,2))}</pre>${r.metadata_missing?'':`<a href="/api/attempts/${encodeURIComponent(r.id)}/files/metadata.json" target="_blank" rel="noopener">metadata.json ↗</a>`}</details></td></tr>`;
  }).join('');
}
function populateClusterFilter() {
  const select=$('#attempt-cluster'),current=select.value;
  const clusters=(results.clusters??[]).filter(c=>keyOfCluster(c)===$('#benchmark-year').value);
  select.innerHTML='<option value="all">All configuration families</option>'+clusters.map(c=>`<option value="${esc(c.id)}">${esc(c.label)} · ${c.count} attempt${c.count===1?'':'s'}</option>`).join('');
  select.value=clusters.some(c=>c.id===current)?current:'all';
  return clusters;
}
function clusterTable(clusters, rows) {
  const byId=new Map(rows.map(r=>[r.id,r]));
  const range=s=>s.n?`${sec(s.min_s)}–${sec(s.max_s)}`:'—';
  return `<div class="table-scroll"><table class="results-table cluster-table"><thead><tr><th>Configuration family</th><th>Reached / attempts</th><th>Median time to 18</th><th>Range</th><th>Matched settings &amp; individual attempts</th></tr></thead><tbody>${clusters.map(c=>{
    const first=byId.get(c.attempt_ids[0]),groups=c.replications.map(g=>{
      const member=byId.get(g.attempt_ids[0]),controls=member?.cluster.recorded_controls??{};
      const benchmark=controls['config.benchmark'],reuse=controls['config.reuse_server'];
      const source=controls['runner.git_commit'];
      return `<details><summary>${g.count} attempt${g.count===1?'':'s'} · ${esc(controls['runner.version']??'runner unavailable')} · seed ${esc(controls['config.seed']??'unrecorded')}</summary><small>${benchmark==null?'Benchmark flag unrecorded':benchmark?'Benchmark mode':'Profiling configuration'} · ${reuse==null?'Server reuse unrecorded':reuse?'Reused server':'Managed server'}<br>Source: ${esc(source??'unrecorded')}</small><ul>${g.attempt_ids.map(id=>{const r=byId.get(id);return `<li><a href="${attemptLink({id})}">${esc(r?.metadata.label??id)}</a><small>${esc(id)} · ${sec(r?.time_to_18_s)} · ${esc(r?.status??'unavailable')}<br>Initial latency: ${sec(r?.first_grader_request_s)}</small></li>`;}).join('')}</ul></details>`;
    }).join('');
    const variations=c.varying_controls.length?`<details><summary>Changed controls within this family (${c.varying_controls.length})</summary><pre>${esc(c.varying_controls.map(v=>`${v.variable}: ${v.values.map(x=>JSON.stringify(x)).join(' / ')}`).join('\n'))}</pre></details>`:'<small>Recorded settings match across this family.</small>';
    return `<tr><td><span class="cluster-swatch" style="background:${modelStyle(first??{}).color}"></span><strong>${esc(c.label)}</strong><small>${c.replications.length} matched-setting group${c.replications.length===1?'':'s'}</small>${variations}</td><td data-label="Reached / attempts">${c.time_to_18.n} / ${c.count}<small>${esc(Object.entries(c.statuses).map(([status,n])=>`${n} ${status}`).join(' · '))}</small></td><td data-label="Median time to 18"><span class="result-time">${sec(c.time_to_18.median_s)}</span><small>Initial median: ${sec(c.initial_latency.median_s)} (${c.initial_latency.n} timed)</small></td><td data-label="Range">${range(c.time_to_18)}</td><td data-label="Matched settings &amp; individual attempts">${groups}</td></tr>`;
  }).join('')}</tbody></table></div>`;
}
function renderClusters(clusters, rows) {
  const selected=$('#attempt-cluster').value;
  const visible=selected==='all'?clusters:clusters.filter(c=>c.id===selected);
  $('#cluster-count').textContent=`${visible.length} configuration ${visible.length===1?'family':'families'} · ${visible.reduce((n,c)=>n+c.count,0)} attempts`;
  const repeated=visible.filter(c=>c.count>1),single=visible.filter(c=>c.count===1);
  $('#cluster-summary').innerHTML=(repeated.length?clusterTable(repeated,rows):'')+(single.length?(selected==='all'?`<details class="singleton-clusters"><summary>${single.length} single-attempt configurations</summary>${clusterTable(single,rows)}</details>`:clusterTable(single,rows)):'')||'<p class="detail-note">No valid configurations available for clustering.</p>';
}
function renderResults(){
    populateBenchmarkFilter();
    const clusters=populateClusterFilter();
    const numberingRows=benchmarkRowsByKey(results.attempts,$('#benchmark-year').value);
    const rows=selectedRows(),ranked=rows.filter(r=>r.time_to_18_s!=null).sort((a,b)=>a.time_to_18_s-b.time_to_18_s),best=ranked[0];
    $('#inventory').textContent=`${rows.length} saved attempts · ${ranked.length} measured targets reached`;
    $('#notice').textContent=results.warnings.join(' · ');$('#notice').hidden=!results.warnings.length;
    const metrics=[['Fastest observed',best?sec(best.time_to_18_s):'—',best?.metadata.label??'No measured target'],['Grader floor','54.00s','3s × 18 correct questions'],['Above the floor',best?sec(best.time_to_18_s-results.reference_floor_s):'—','Fastest run, after warmup']];
    $('#headline-metrics').innerHTML=metrics.map(([title,value,note],i)=>`<article class="metric-card ${i===0?'score-card':''}"><div class="metric-label">${title}</div><div class="metric-value">${value}</div><div class="metric-sub">${esc(note)}</div></article>`).join('');
    const history=attemptHistory(rows,numberingRows);
    $('#history-model-legend').innerHTML=modelLegend(history.points);
    $('#comparison-plot').innerHTML=comparisonPlot(history);$('#history-legend').innerHTML=historyLegend(history);$('#floor-note').textContent=results.floor_note;
    $('#baseline-note').innerHTML=history.baselines.length?`Separate baselines excluded from this plot: ${history.baselines.map(r=>`<a href="${attemptLink(r)}">${r.plot_number}. ${esc(r.metadata.label)}</a> (${sec(r.time_to_18_s)})`).join(' · ')}. Their records remain in the full attempt table below.`:'';
    const initial=initialLatencyHistory(rows,numberingRows);
    $('#initial-model-legend').innerHTML=modelLegend(initial.points);
    $('#initial-plot').innerHTML=initialLatencyPlot(initial);$('#initial-legend').innerHTML=initialLatencyLegend(initial);
    $('#initial-unavailable').innerHTML=initial.unavailable.length?`Request timing unavailable: ${initial.unavailable.map(r=>`<a href="${attemptLink(r)}">${esc(r.metadata?.label??r.id)}</a> (${esc(r.status)})`).join(' · ')}.`:'';
    $('#excluded-note').textContent=rows.filter(r=>r.time_to_18_s==null || !Number.isFinite(Date.parse(r.attempt_started_at_utc??r.started_at_utc))).map(r=>`${r.metadata?.label??r.id}: ${r.time_to_18_s==null?r.status:'start timestamp unavailable'}${r.solved!=null?` (${r.solved} correct)`:''}`).join(' · ');
    renderClusters(clusters,numberingRows);
    $('#interventions').innerHTML=interventionCards(rows);$('#attempt-rows').innerHTML=controlsTable(rows);renderProgress();$('#content').hidden=false;
}
async function refresh(){
  $('#refresh').disabled=true;$('#error').hidden=true;
  try{
    const response=await fetch('/api/results',{cache:'no-store'});if(!response.ok)throw new Error(`Unable to load results (${response.status})`);results=await response.json();
    renderResults();
  }catch(e){$('#error').textContent=e.message;$('#error').hidden=false;}finally{$('#refresh').disabled=false;}
}
document.addEventListener('DOMContentLoaded',()=>{$('#refresh').addEventListener('click',refresh);$('#benchmark-year').addEventListener('change',()=>{if(results)renderResults();});$('#attempt-cluster').addEventListener('change',()=>{if(results)renderResults();});$('#curve-window').addEventListener('change',renderProgress);refresh();});
