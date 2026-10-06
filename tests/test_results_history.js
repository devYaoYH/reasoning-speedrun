/**
 * Offline checks for the results plot's chronological Pareto frontier.
 * Run `node tests/test_results_history.js` from the repository root. The VM loads
 * the real viewer code without a browser or network; tests cover non-improving
 * attempts, ties, invalid evidence, initialization timestamps, and SVG axes.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const context = vm.createContext({document:{addEventListener(){}}});
vm.runInContext(fs.readFileSync(path.join(__dirname,'../src/reasoning_speedrun/viewer/results_ui/viewer.js'),'utf8'),context);
function row(id, start, latency, official=start) {
  return {id,attempt_started_at_utc:start,started_at_utc:official,time_to_18_s:latency,
    metadata:{label:id,intervention:{label:'test intervention'}}};
}
function history(rows) {
  context.rows=rows;
  return JSON.parse(JSON.stringify(vm.runInContext('attemptHistory(rows)',context)));
}
const rows=[row('slow','2026-10-03T20:40:00Z',300),
  row('first','2026-10-03T20:00:00Z',100,'2026-10-03T20:59:00Z'),
  row('best','2026-10-03T20:30:00Z',70),row('tie','2026-10-03T20:31:00Z',70),
  row('unmet','2026-10-03T20:50:00Z',null),row('missing','bad timestamp',60),
  row('invalid','2026-10-03T20:10:00Z',NaN),row('negative','2026-10-03T20:15:00Z',-1)];
const before=rows.map(r=>r.id);
const result=history(rows);
assert.deepEqual(result.points.map(r=>r.id),['first','best','tie','slow']);
assert.deepEqual(result.frontier.map(r=>r.id),['first','best']);
assert.deepEqual(rows.map(r=>r.id),before,'Sorting must not mutate the inventory');
assert.deepEqual(history([]),{points:[],frontier:[],baselines:[]});
assert.equal(history([row('legacy',null,54,'2026-10-03T20:00:00Z')]).points.length,1);
context.rows=rows;
vm.runInContext('results={reference_floor_s:54}',context);
const svg=vm.runInContext('comparisonPlot(attemptHistory(rows))',context);
assert.equal((svg.match(/class="attempt-dot"/g)||[]).length,4);
assert.match(svg,/class="pareto-frontier" d="M [^"]+ H [^"]+ V [^"]+ H [^"]+"/);
const floor=svg.match(/class="floor-line" x1="([^"]+)" x2="([^"]+)" y1="([^"]+)" y2="([^"]+)"/);
assert.ok(floor);assert.notEqual(floor[1],floor[2]);assert.equal(floor[3],floor[4]);
assert.match(svg,/Attempt started · America\/Los_Angeles/);
assert.match(svg,/Time to 18 verified correct answers \(seconds\)/);
context.rows=[row('single','2026-10-03T20:00:00Z',54)];
assert.doesNotMatch(vm.runInContext('comparisonPlot(attemptHistory(rows))',context),/NaN|Infinity/);
context.rows=[row('first','2026-10-03T20:07:00Z',92),
  row('20261003T202152.418590Z','2026-10-03T20:21:00Z',336),
  row('20261003T203338.063138Z','2026-10-03T20:33:00Z',517),
  row('improved','2026-10-03T20:53:00Z',85),row('retry','2026-10-03T21:02:00Z',97)];
const filtered=history(context.rows);
assert.deepEqual(filtered.points.map(r=>r.plot_number),[1,4,5]);
assert.deepEqual(filtered.baselines.map(r=>r.plot_number),[2,3]);
assert.deepEqual(filtered.frontier.map(r=>r.id),['first','improved']);
const filteredSvg=vm.runInContext('comparisonPlot(attemptHistory(rows))',context);
assert.equal((filteredSvg.match(/class="attempt-dot"/g)||[]).length,3);
assert.doesNotMatch(filteredSvg,/20261003T202152|20261003T203338/);
assert.match(filteredSvg,/>100s<\/text>/,'Y-axis must rescale around the remaining runs');
assert.match(filteredSvg,/>50s<\/text>/,'Y-axis must start at 50 seconds');
assert.doesNotMatch(filteredSvg,/>0s<\/text>/);
const filteredFloor=filteredSvg.match(/class="floor-line"[^>]* y1="([^"]+)"/);
assert.ok(Math.abs(Number(filteredFloor[1])-(330-4/50*290))<1e-9,'54-second floor must use the shifted scale');
context.rows=[
  {...row('20261003T202152.418590Z','2026-10-03T20:21:00Z',336),first_grader_request_s:30},
  {...row('interrupted','2026-10-03T20:20:00Z',null),first_grader_request_s:0},
  {...row('missing','2026-10-03T20:22:00Z',92),first_grader_request_s:null},
  {...row('invalid','bad timestamp',92),first_grader_request_s:5},
  {...row('negative','2026-10-03T20:23:00Z',92),first_grader_request_s:-1}];
const initial=JSON.parse(JSON.stringify(vm.runInContext('initialLatencyHistory(rows)',context)));
assert.deepEqual(initial.points.map(r=>r.id),['interrupted','20261003T202152.418590Z']);
assert.deepEqual(initial.points.map(r=>r.initial_number),[1,2]);
assert.equal(initial.unavailable.length,3);
const initialSvg=vm.runInContext('initialLatencyPlot(initialLatencyHistory(rows))',context);
assert.equal((initialSvg.match(/class="attempt-dot initial-dot"/g)||[]).length,2);
assert.match(initialSvg,/>0s<\/text>/);
assert.match(initialSvg,/>A1<\/text>/);
assert.match(initialSvg,/Time to first grader request \(seconds\)/);
assert.doesNotMatch(initialSvg,/floor-line|pareto-frontier|NaN|Infinity/);
context.rows=[];
assert.match(vm.runInContext('initialLatencyPlot(initialLatencyHistory(rows))',context),/No recorded first grader request/);
const variants=[
  {id:'WeiboAI/VibeThinker-3B',quantization:'none',activation_dtype:'bfloat16'},
  {id:'r0b0tlab/VibeThinker-3B-NVFP4',quantization:'modelopt_fp4',activation_dtype:'bfloat16'},
  {id:'Qwen/Qwen3.5-4B',quantization:'none',activation_dtype:'bfloat16'},
  {id:'Qwen/Qwen3.5-35B-A3B-GPTQ-Int4',quantization:'GPTQ Int4',activation_dtype:'bfloat16'}];
context.colorRows=[variants[0],variants[1],variants[0],variants[2],variants[3]].map((model,i)=>{
  const r=row(`color-${i}`,`2026-10-03T20:0${i}:00Z`,90-i*5);
  return {...r,metadata:{...r.metadata,model},first_grader_request_s:10-i,
    events:[{elapsed_s:20}],settlement_s:100};
});
const styles=JSON.parse(JSON.stringify(vm.runInContext('colorRows.map(modelStyle)',context)));
assert.equal(styles[0].color,styles[2].color,'Repeated BF16 runs must share a model color');
assert.equal(new Set(styles.map(s=>s.color)).size,4,'All four model variants must have distinct colors');
context.changedQuant={metadata:{model:{...variants[0],quantization:'modelopt_fp4'}}};
assert.notEqual(vm.runInContext('modelStyle(changedQuant).color',context),styles[0].color,
  'Quantization must remain part of model identity even under the same model ID');
const colorMap=svg=>Object.fromEntries([...svg.matchAll(/<circle[^>]*data-attempt="([^"]+)"[^>]*fill="([^"]+)"/g)].map(m=>[m[1],m[2]]));
const targetColors=colorMap(vm.runInContext('comparisonPlot(attemptHistory(colorRows))',context));
const initialColors=colorMap(vm.runInContext('initialLatencyPlot(initialLatencyHistory([...colorRows].reverse()))',context));
assert.deepEqual(initialColors,targetColors,'Colors must match across plots and inventory order');
const key=vm.runInContext('modelLegend(colorRows)',context);
assert.equal((key.match(/class="model-key"/g)||[]).length,4);
assert.match(key,/VibeThinker 3B · BF16/);assert.match(key,/VibeThinker 3B · NVFP4/);
const elements={'#benchmark-year':{value:'2025'},'#attempt-cluster':{value:'all'},'#curve-window':{value:'all'},
  '#progress-plot':{},'#progress-model-legend':{},'#curve-legend':{}};
context.document.querySelector=selector=>elements[selector];
vm.runInContext('results={attempts:colorRows};renderProgress()',context);
const progressColors=Object.fromEntries([...elements['#progress-plot'].innerHTML.matchAll(/<path[^>]*stroke="([^"]+)"[^>]*><title>([^<]+)<\/title>/g)].map(m=>[m[2],m[1]]));
assert.deepEqual(progressColors,targetColors,'Progress curves must use the same model colors');
assert.equal(elements['#progress-model-legend'].innerHTML,key);
context.subset=[context.colorRows[2],context.colorRows[4]];
const numbered=JSON.parse(JSON.stringify(vm.runInContext('attemptHistory(subset,colorRows)',context)));
assert.deepEqual(numbered.points.map(r=>r.plot_number),[3,5],'Cluster filters must preserve original point numbers');
const initialNumbered=JSON.parse(JSON.stringify(vm.runInContext('initialLatencyHistory(subset,colorRows)',context)));
assert.deepEqual(initialNumbered.points.map(r=>r.initial_number),[3,5]);
context.familyRows=[{id:'a',cluster:{family_id:'one'}},{id:'b',cluster:{family_id:'two'}},{id:'invalid'}];
assert.deepEqual(JSON.parse(JSON.stringify(vm.runInContext('clusterRows(familyRows,"one")',context))).map(r=>r.id),['a']);
assert.equal(vm.runInContext('clusterRows(familyRows,"all").length',context),3);
console.log('Results history checks passed: chronological points, strict running minimum, timestamps, floor, initial latency, model colors, and SVG.');
context.mixedYears=[{id:'dev',benchmark_year:2025},{id:'test',benchmark_year:2026},{id:'legacy'}];
assert.deepEqual(JSON.parse(JSON.stringify(vm.runInContext('benchmarkRows(mixedYears, 2026)',context))).map(r=>r.id),['test']);
assert.deepEqual(JSON.parse(JSON.stringify(vm.runInContext('benchmarkRows(mixedYears, 2025)',context))).map(r=>r.id),['dev','legacy']);

context.warmupYears=[{id:'warmup',benchmark_year:2024},{id:'dev',benchmark_year:2025},{id:'test',benchmark_year:2026}];
assert.deepEqual(JSON.parse(JSON.stringify(vm.runInContext('benchmarkRows(warmupYears, 2024)',context))).map(r=>r.id),['warmup']);

// Generic datasets have an explicit null year and must never enter AIME charts.
context.transferRows=[{id:'legacy'}, {id:'aime',benchmark_year:2025},
  {id:'apex',benchmark_year:null,metadata:{provenance:{dataset:{id:'apex_shortlist',year:null}}}}];
assert.deepEqual(JSON.parse(JSON.stringify(vm.runInContext("benchmarkRows(transferRows,2025).map(r=>r.id)",context))),['legacy','aime']);
