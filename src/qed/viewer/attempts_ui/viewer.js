/**
 * Inspect canonical src.attempt outputs: oracle outcomes, saved trajectories,
 * continuation segments, provenance, and sampled GPU telemetry. Use via
 * src.viewer_server; refresh rereads local files without running experiments.
 */
const $ = selector => document.querySelector(selector);
const state = { id: null, overview: null, question: null, rollout: null, detail: null,
  trajectory: null, tab: 'reasoning', filter: 'all', search: '', sequence: 0, detailSequence: 0, busy: false };
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const number = value => value == null ? '—' : new Intl.NumberFormat('en-US').format(value);
const seconds = value => value == null ? '—' : `${Number(value).toFixed(2)}s`;
const duration = value => value == null ? '—' : `${Math.floor(Math.round(value) / 60)}m ${String(Math.round(value) % 60).padStart(2, '0')}s`;
const median = values => { const sorted = values.filter(v => v != null).sort((a, b) => a - b); const n = sorted.length; return n ? (sorted[Math.floor(n / 2)] + sorted[Math.floor((n - 1) / 2)]) / 2 : null; };
const elapsed = timestamp => timestamp && state.overview.attempt.started_at_utc ? (new Date(timestamp) - new Date(state.overview.attempt.started_at_utc)) / 1000 : null;
const api = suffix => `/api/attempts/${encodeURIComponent(state.id)}/${suffix}`;
const artifact = relative => api(`files/${relative}`);

async function getJson(url) {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${response.status} while loading ${url}`);
  return response.json();
}
function error(message) { $('#error').textContent = message; $('#error').hidden = !message; }
function notice(message) { $('#notice').textContent = message; $('#notice').hidden = !message; }
function updateUrl() {
  const url = new URL(location.href);
  url.searchParams.delete('run');
  url.searchParams.set('attempt', state.id);
  if (state.question != null) url.searchParams.set('q', state.question);
  else url.searchParams.delete('q');
  if (state.rollout != null) url.searchParams.set('rollout', state.rollout);
  else url.searchParams.delete('rollout');
  history.replaceState(null, '', url);
}

async function refresh(requested = state.id) {
  if (state.busy) return;
  state.busy = true;
  const sequence = ++state.sequence;
  ++state.detailSequence;
  $('#refresh').disabled = true;
  $('#attempt-select').disabled = true;
  error('');
  try {
    const inventory = await getJson('/api/attempts');
    const select = $('#attempt-select');
    select.replaceChildren();
    for (const item of inventory.attempts) {
      const option = document.createElement('option');
      option.value = item.id;
      option.textContent = `${item.id} · AIME ${item.benchmark_year ?? 2025} · ${item.model || 'unknown model'} · ${item.status} · ${number(item.solved)}/${number(item.questions)} solved`;
      select.append(option);
    }
    $('#empty').hidden = inventory.attempts.length > 0;
    $('#content').hidden = !inventory.attempts.length;
    select.disabled = !inventory.attempts.length;
    if (!inventory.attempts.length) { notice(inventory.warnings.join(' · ')); return; }
    const selected = inventory.attempts.find(a => a.id === requested) || inventory.attempts.find(a => a.trace_questions > 0) || inventory.attempts[0];
    if (state.id !== selected.id) { state.question = null; state.rollout = null; }
    state.id = selected.id;
    select.value = state.id;
    const [overviewResult, gpuResult] = await Promise.allSettled([getJson(api('overview')), getJson(api('gpu'))]);
    if (sequence !== state.sequence) return;
    if (overviewResult.status !== 'fulfilled') throw overviewResult.reason;
    state.overview = overviewResult.value;
    const warnings = [...inventory.warnings];
    if (!state.overview.summary) warnings.push('No final summary is present. This may be an active attempt or an incomplete local copy.');
    if (!state.overview.questions.some(q => q.trace_available)) warnings.push('Only compact results are available locally. Copy trace/ from this attempt to inspect trajectories.');
    if (state.overview.summary?.error) warnings.push(state.overview.summary.error);
    if (gpuResult.status === 'rejected') warnings.push(`GPU telemetry unavailable: ${gpuResult.reason.message}`);
    notice(warnings.join(' · '));
    renderOverview(); renderCurve(); renderGpu(gpuResult.status === 'fulfilled' ? gpuResult.value : null); renderQuestions();
    const params = new URLSearchParams(location.search);
    const question = state.overview.questions.find(q => q.problem_idx === (state.question ?? Number(params.get('q')))) || state.overview.questions[0];
    if (question) await selectQuestion(question.problem_idx, state.rollout ?? (Number(params.get('rollout')) || null));
    else { state.question = null; state.rollout = null; $('#detail').textContent = 'No question outcomes have been recorded.'; updateUrl(); }
  } catch (exc) { error(exc.message); }
  finally { state.busy = false; $('#refresh').disabled = false; $('#attempt-select').disabled = !$('#attempt-select').options.length; }
}

function renderOverview() {
  const { attempt, summary, config, questions } = state.overview;
  const rollouts = questions.flatMap(q => q.rollouts);
  const usage = rollouts.filter(r => r.usage?.completion_tokens != null);
  const solved = summary?.solved ?? questions.filter(q => q.status === 'solved').length;
  const queries = questions.filter(q => q.verification_count != null);
  $('#model-name').textContent = `AIME ${state.overview.attempt.benchmark_year ?? 2025} · ${config.model || 'Unknown model'}`;
  $('#attempt-date').textContent = attempt.started_at_utc ? new Date(attempt.started_at_utc).toLocaleString() : 'No official start recorded';
  $('#attempt-status').textContent = `${attempt.status} · ${summary?.strategy || config.strategy || 'parallel'} strategy`;
  $('#overview-title').textContent = `${solved} of ${questions.length} verified`;
  $('#overview-note').textContent = `${config.parallelism ?? '—'} concurrent questions · ${config.rollouts ?? '—'} ${config.rollouts === 1 ? 'rollout' : 'rollouts'} per round · ${number(config.max_tokens)} token ceiling · ${config.disable_thinking ? 'thinking disabled' : 'model-default thinking'}`;
  const cards = [
    ['Verified solutions', `${solved} / ${questions.length}`, summary?.target_correct ? `Target ${summary.target_correct}: ${summary.target_reached ? 'reached' : 'not reached'}` : 'Positive oracle verdicts'],
    ['Official time', duration(summary?.official_latency_s), 'After warmup through attempt settlement'],
    ['Generation requests', rollouts.length ? number(rollouts.length) : '—', 'Includes continuation segments'],
    ['Recorded output tokens', usage.length ? number(usage.reduce((n, r) => n + r.usage.completion_tokens, 0)) : '—', `Usage present for ${usage.length}/${rollouts.length} local rollouts`],
    ['Verification queries', queries.length ? number(queries.reduce((n, q) => n + q.verification_count, 0)) : '—', `${config.grader_cost ?? '—'}s configured global toll · ${queries.length}/${questions.length} question logs`],
    ['Median observed TTFT', seconds(median(rollouts.map(r => r.ttft_s))), 'Queue + prefill + first received output delta'],
  ];
  $('#metrics').innerHTML = cards.map(([title, value, note], i) => `<article class="metric-card ${i === 0 ? 'score-card' : ''}"><div class="metric-label">${esc(title)}</div><div class="metric-value">${esc(value)}</div><div class="metric-sub">${esc(note)}</div></article>`).join('');
  $('#config').textContent = JSON.stringify({ experiment_metadata: state.overview.experiment_metadata, config, summary }, null, 2);
  $('#file-links').innerHTML = ['config.json', ...(summary ? ['summary.json'] : []), ...(state.overview.experiment_metadata ? ['metadata.json'] : [])].map(file => `<a target="_blank" rel="noopener" href="${artifact(file)}">${file} ↗</a>`).join('');
}

function chartFrame(xmax, ymax, yLabel, paths, secondary = false) {
  const width = 960, height = 255, left = 58, right = secondary ? 65 : 20, top = 25, bottom = 35;
  const plotW = width - left - right, plotH = height - top - bottom;
  const x = v => left + v / Math.max(xmax, 1) * plotW;
  const y = v => top + plotH - v / Math.max(ymax, 1) * plotH;
  let svg = `<svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(yLabel)} over elapsed seconds"><text x="${left}" y="15">${esc(yLabel)}</text>`;
  for (let i = 0; i <= 4; i++) {
    const px = x(xmax * i / 4), py = y(ymax * i / 4);
    svg += `<line class="grid" x1="${left}" x2="${width - right}" y1="${py}" y2="${py}"/><text text-anchor="end" x="${left - 10}" y="${py + 4}">${Number((ymax * i / 4).toFixed(1))}</text><text text-anchor="middle" x="${px}" y="${height - 10}">${duration(xmax * i / 4)}</text>`;
    if (secondary) svg += `<text x="${width - right + 12}" y="${py + 4}">${i * 25}%</text>`;
  }
  svg += paths(x, y) + '</svg>';
  return svg;
}
function renderCurve() {
  const { summary, questions, solved_events } = state.overview;
  const byQuestion = new Map();
  for (const event of [...solved_events, ...questions.map(q => q.first_solved).filter(Boolean)]) {
    const time = event.first_solved_elapsed_s ?? elapsed(event.first_solved_at_utc);
    if (time != null && Number.isFinite(time) && time >= 0) {
      const old = byQuestion.get(event.problem_idx);
      if (old == null || time < old) byQuestion.set(event.problem_idx, time);
    }
  }
  const times = [...byQuestion.values()].sort((a, b) => a - b);
  const target = (summary?.strategy || state.overview.config.strategy) === 'coverage'
    ? summary?.target_correct ?? state.overview.config.target_correct : null;
  $('#target-note').textContent = target ? `Target: ${target} questions` : `${times.length} timestamped solutions`;
  if (!times.length) { $('#solve-curve').innerHTML = '<div class="chart-empty">No timestamped positive verdicts are available locally.</div>'; return; }
  const xmax = Math.max(summary?.official_latency_s ?? 0, times.at(-1), 1);
  $('#solve-curve').innerHTML = chartFrame(xmax, questions.length, 'Verified questions', (x, y) => {
    let d = `M ${x(0)} ${y(0)}`;
    times.forEach((time, i) => { d += ` H ${x(time)} V ${y(i + 1)}`; });
    d += ` H ${x(xmax)}`;
    return `<path class="solution" d="${d}"/>${target ? `<line class="target" x1="${x(0)}" x2="${x(xmax)}" y1="${y(target)}" y2="${y(target)}"/>` : ''}`;
  });
}
function renderGpu(data) {
  const rows = (data?.samples || []).map(row => ({ ...row, elapsed: elapsed(row.timestamp_utc) })).filter(row => row.elapsed != null && row.elapsed >= 0 && (state.overview.summary?.official_latency_s == null || row.elapsed <= state.overview.summary.official_latency_s));
  $('#gpu-meta').textContent = data?.sample_count ? `${number(data.sample_count)} total saved samples · green: GiB · amber: utilization` : 'No local samples';
  if (!rows.length) { $('#gpu-plot').innerHTML = '<div class="chart-empty">GPU samples from the official solving phase are unavailable locally.</div>'; return; }
  const xmax = Math.max(state.overview.summary?.official_latency_s ?? 0, ...rows.map(r => r.elapsed), 1);
  const ymax = Math.max(...rows.map(r => (r.vram_total_mib ?? r.vram_used_mib ?? 0) / 1024), 1);
  $('#gpu-plot').innerHTML = chartFrame(xmax, ymax, 'Shared device VRAM (GiB)', (x, y) => {
    const line = (field, convert, cls) => {
      const points = rows.filter(r => r[field] != null).map(r => `${x(r.elapsed)},${y(convert(r[field]))}`).join(' ');
      return `<polyline class="${cls}" points="${points}"/>`;
    };
    return line('vram_used_mib', value => value / 1024, 'memory') + line('gpu_util_pct', value => value / 100 * ymax, 'util');
  }, true);
}

function renderQuestions() {
  const rows = state.overview.questions.filter(q => (state.filter === 'all' || q.status === state.filter) && `${q.problem_idx} ${q.problem}`.toLowerCase().includes(state.search));
  $('#visible-count').textContent = rows.length;
  $('#question-list').innerHTML = rows.map(q => `<button type="button" class="question-item ${q.status === 'error' ? 'error' : ''} ${q.problem_idx === state.question ? 'selected' : ''}" data-question="${q.problem_idx}" aria-label="Question ${q.problem_idx}, ${esc(q.status)}"><span class="question-number">${String(q.problem_idx).padStart(2, '0')}</span><span class="question-copy"><span class="question-preview">${esc(q.problem || 'Problem text unavailable')}</span><span class="question-meta">${esc(q.status)} · ${q.rollouts.length} local rollouts · ${esc(duration(q.end_to_end_latency_s))}${q.trace_available ? '' : ' · summary only'}</span></span></button>`).join('');
}
async function selectQuestion(index, desiredRollout = null) {
  const sequence = ++state.detailSequence;
  state.question = index; state.rollout = null; state.trajectory = null;
  renderQuestions(); updateUrl();
  $('#detail').innerHTML = '<div class="empty-state">Loading saved question…</div>';
  try {
    const detail = await getJson(api(`questions/${index}`));
    if (sequence !== state.detailSequence) return;
    state.detail = detail;
    const q = state.overview.questions.find(row => row.problem_idx === index);
    const chosen = detail.rollouts.find(r => r.rollout === desiredRollout) || detail.rollouts.find(r => r.rollout === q.winning_rollout) || detail.rollouts[0];
    state.rollout = chosen?.rollout ?? null;
    renderDetail(); updateUrl();
    if (chosen) await loadRollout(chosen.rollout, sequence);
  } catch (exc) { if (sequence === state.detailSequence) error(exc.message); }
}
function renderDetail() {
  const q = state.overview.questions.find(row => row.problem_idx === state.question);
  const detail = state.detail;
  const verdicts = detail.verification;
  $('#detail').innerHTML = `<div class="detail-top"><div class="eyebrow">Question ${q.problem_idx} · canonical attempt</div><h2>Question ${q.problem_idx}<span class="badge ${q.status === 'solved' ? 'good' : q.status === 'error' ? 'bad' : 'neutral'}">${esc(q.status)}</span></h2><p>${q.verified_answer != null ? `Verified answer <strong>${esc(q.verified_answer)}</strong> · winning rollout ${esc(q.winning_rollout)}` : 'No positive oracle verdict recorded.'}</p><p class="detail-note">Question settlement: ${esc(seconds(q.end_to_end_latency_s))} · ${number(q.unique_candidates)} distinct candidates${q.first_solved ? ` · first solved +${esc(seconds(q.first_solved.first_solved_elapsed_s ?? elapsed(q.first_solved.first_solved_at_utc)))}` : ''}</p>${q.error ? `<div class="notice">${esc(q.error)}</div>` : ''}<h3>Problem statement</h3><pre class="problem-text">${esc(q.problem)}</pre>${detail.trace_available ? '' : '<div class="notice">This question has summary statistics only. Copy its trace directory locally to inspect saved responses and verifications.</div>'}<h3>Oracle verification</h3>${verdicts.length ? `<table class="verifications"><thead><tr><th>Candidate</th><th>Rollout / round</th><th>Verdict</th><th>Observed at</th><th>Check latency</th></tr></thead><tbody>${verdicts.map(v => `<tr><td>${esc(v.candidate)}</td><td>${esc(v.rollout)} / ${esc(v.round ?? '—')}</td><td>${v.error ? esc(v.error) : v.result?.verdict === true ? 'Correct' : v.result?.verdict === false ? 'Incorrect' : 'Unknown'}</td><td>+${esc(seconds(elapsed(v.observed_at_utc)))}</td><td>${esc(seconds(v.verification_latency_s))}</td></tr>`).join('')}</tbody></table>` : `<p class="detail-note">${detail.trace_available ? 'No completed verification events in the local log.' : 'Verification log unavailable locally.'}</p>`}<h3>Generation rollouts</h3><div class="rollout-buttons">${detail.rollouts.map(r => `<button type="button" data-rollout="${r.rollout}" class="${r.rollout === state.rollout ? 'selected' : ''}">Rollout ${r.rollout}<small>Round ${r.round ?? '—'} · ${esc(r.status || 'metadata pending')}${r.continuation_of_rollout != null ? ` · continues ${r.continuation_of_rollout}` : ''}</small></button>`).join('')}</div><div id="rollout-detail">${detail.rollouts.length ? 'Loading trajectory…' : '<p class="detail-note">No rollout artifacts are available locally.</p>'}</div></div>`;
  $('.rollout-buttons').addEventListener('click', event => {
    const button = event.target.closest('[data-rollout]');
    if (button) { state.rollout = Number(button.dataset.rollout); state.trajectory = null; renderDetail(); updateUrl(); loadRollout(state.rollout, ++state.detailSequence); }
  });
}
async function loadRollout(rolloutNumber, sequence) {
  try {
    const result = await getJson(api(`questions/${state.question}/rollouts/${rolloutNumber}`));
    if (sequence !== state.detailSequence) return;
    state.trajectory = result;
    const meta = { ...state.detail.rollouts.find(r => r.rollout === rolloutNumber), ...result.telemetry };
    const request = result.request;
    const recordedProblem = request?.messages?.find(m => m.role === 'user')?.content;
    if (recordedProblem) $('#detail .problem-text').textContent = recordedProblem;
    const censored = meta.generation_censored ?? (meta.status === 'cancelled' || meta.finish_reason === 'length');
    $('#rollout-detail').innerHTML = `<div class="rollout-stats">${[['TTFT', seconds(meta.ttft_s)], ['Observed generation', seconds(meta.generation_latency_s)], ['Through settlement', seconds(meta.end_to_end_latency_s)], ['Output tokens', number(meta.usage?.completion_tokens)], ['Cached prompt tokens', number(meta.cached_prompt_tokens)], ['Sampled shared peak VRAM', meta.gpu?.observed_peak_vram_mib != null ? `${(meta.gpu.observed_peak_vram_mib / 1024).toFixed(2)} GiB` : '—']].map(([title, value]) => `<div><small>${esc(title)}</small><strong>${esc(value)}</strong></div>`).join('')}</div><p class="detail-note">${esc(meta.status || 'Metadata pending')} · finish reason ${esc(meta.finish_reason ?? '—')} · ${esc(meta.endpoint || 'endpoint unavailable')}${meta.continuation_of_rollout != null ? ` · continues rollout ${meta.continuation_of_rollout}` : ''}${censored ? ' · censored: cancelled or token-capped; observed duration is not natural completion time' : ''}</p>${meta.error ? `<div class="notice">${esc(meta.error)}</div>` : ''}<div class="artifact-links">${result.files.map(file => `<a href="${artifact(`trace/${String(state.question).padStart(2, '0')}/rollout-${String(rolloutNumber).padStart(2, '0')}/${file}`)}" target="_blank" rel="noopener">${esc(file)} ↗</a>`).join('')}</div><div class="trace-tabs" role="tablist" aria-label="Trajectory content">${[['reasoning', 'Reasoning'], ['content', 'Response'], ['request', 'Request'], ['telemetry', 'Telemetry'], ['verification', 'Verification events']].map(([key, title]) => `<button type="button" role="tab" data-tab="${key}" aria-selected="${key === state.tab}" class="${key === state.tab ? 'active' : ''}">${title}</button>`).join('')}</div><pre id="trajectory-text" class="trajectory-text" role="tabpanel"></pre>`;
    $('.trace-tabs').addEventListener('click', event => {
      const button = event.target.closest('[data-tab]');
      if (button) { state.tab = button.dataset.tab; renderText(); }
    });
    renderText();
  } catch (exc) { if (sequence === state.detailSequence) $('#rollout-detail').textContent = `Trajectory unavailable: ${exc.message}`; }
}
function renderText() {
  const result = state.trajectory;
  document.querySelectorAll('[data-tab]').forEach(button => { button.classList.toggle('active', button.dataset.tab === state.tab); button.setAttribute('aria-selected', button.dataset.tab === state.tab); });
  const value = state.tab === 'verification' ? state.detail.verification : state.tab === 'request' ? result.request : state.tab === 'telemetry' ? result.telemetry : result.response?.[state.tab];
  $('#trajectory-text').textContent = value == null ? 'This artifact is unavailable locally.' : typeof value === 'string' ? value || `No ${state.tab} text was saved for this rollout.` : JSON.stringify(value, null, 2);
}

document.addEventListener('DOMContentLoaded', () => {
  $('#refresh').addEventListener('click', () => refresh());
  $('#attempt-select').addEventListener('change', event => refresh(event.target.value));
  $('#search').addEventListener('input', event => { state.search = event.target.value.trim().toLowerCase(); if (state.overview) renderQuestions(); });
  $('#filters').addEventListener('click', event => {
    const button = event.target.closest('[data-filter]');
    if (!button) return;
    state.filter = button.dataset.filter;
    document.querySelectorAll('[data-filter]').forEach(node => node.classList.toggle('active', node === button));
    if (state.overview) renderQuestions();
  });
  $('#question-list').addEventListener('click', event => { const button = event.target.closest('[data-question]'); if (button) selectQuestion(Number(button.dataset.question)); });
  setInterval(() => { if ($('#auto-refresh').checked) refresh(); }, 5000);
  refresh(new URLSearchParams(location.search).get('attempt'));
});
