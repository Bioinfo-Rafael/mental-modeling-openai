// Embedded by build_results_viewer.py. All charts use the frozen, local run data.
const choiceOrder = ['NONE', 'A2', 'A1', 'BOTH'];
const choiceLabels = {NONE: 'No', A2: 'A2', A1: 'A1', BOTH: 'Both'};
const choiceColors = {NONE: '#64748b', A2: '#cf6b15', A1: '#1769aa', BOTH: '#8b3db3'};
const apiLabels = {decisions: 'Decisions', responses_diagnostic: 'Responses'};
let analysisRepeat = '1';
let latencyScale = 'log';

function responseText(record) {
    const parts = (record?.raw?.output || []).flatMap(item => item.content || []);
    return parts.map(p => p.type === 'output_text' ? p.text : p.type === 'refusal' ? p.refusal : '')
        .filter(Boolean).join('\n') || record?.error || '回答本文なし';
}

function hasChoice(row) {
    return row?.status === 'ok' && choiceOrder.includes(row.aircraft_choice);
}

// No ordinal distance: A1 versus A2 differs on both aircraft (distance 2).
function aircraftDistance(a, b) {
    if (!choiceOrder.includes(a) || !choiceOrder.includes(b)) return null;
    const bits = choiceOrder.indexOf(a) ^ choiceOrder.indexOf(b);
    return (bits & 1) + ((bits >> 1) & 1);
}

function analysisLegend(series) {
    return series.map(s => `<span class="series-key"><i aria-hidden="true" class="${s.square ? 'legend-square' : 'legend-dot'}" style="--series-color:${s.color}"></i>${esc(s.name)}</span>`).join('');
}

function pointTitle(row, value) {
    return `${apiLabels[row.condition] || row.condition} r${row.repeat} · 入力 ${row.start_step}–${row.end_step}\n注釈 ${choiceLabels[row.expected_aircraft_choice]} / 回答 ${choiceLabels[row.aircraft_choice] || statuses[row.status]}\n${value}`;
}

// Each series has one entry per planned window. Null values break connecting lines.
function analysisPlot({title, rows, series, ticks, maxY, guide = null, statusRows = [], scale = 'linear', controls = ''}) {
    if (!rows.length) return '<p class="muted">この条件はありません。</p>';
    const W = 1120, H = 270, L = 90, R = 24, T = 25, B = 65;
    const ends = rows.map(r => r.end_step), minX = Math.min(...ends), maxX = Math.max(...ends);
    const x = step => L + (step - minX) / (maxX - minX || 1) * (W - L - R);
    const transform = value => scale === 'log' ? Math.log1p(value) : value;
    const y = value => T + (1 - transform(value) / transform(maxY)) * (H - T - B);
    const annotations = [...new Map(rows.flatMap(row => row.expected_events || [])
        .map(event => [event.id || `${event.aircraft_id}@${event.step}`, event])).values()];
    const annotationLines = annotations.filter(event => event.step >= minX && event.step <= maxX)
        .map(event => `<line class="annotation-line" data-step="${event.step}" x1="${x(event.step)}" x2="${x(event.step)}" y1="${T}" y2="${H-B}" stroke="#dc5555" stroke-opacity=".3" stroke-width="1.5"><title>${esc(`注釈 step ${event.step} · ${event.aircraft_id} · ${event.label || ''}`)}</title></line>`).join('');
    const grid = ticks.map(t => `<line x1="${L}" x2="${W-R}" y1="${y(t.value)}" y2="${y(t.value)}" stroke="#dce4ee"/><text x="${L-10}" y="${y(t.value)+4}" text-anchor="end" font-size="13">${esc(t.label)}</text>`).join('');
    const xTicks = [...new Set([minX, ...Array.from({length: 7}, (_, i) => Math.round(minX + i * (maxX-minX)/6)), maxX])];
    const axis = xTicks.map(step => `<text x="${x(step)}" y="${H-23}" text-anchor="middle" font-size="12">${step}</text>`).join('');
    function path(points, stepwise = false) {
        let result = '', previous = null;
        for (const p of points) {
            if (!Number.isFinite(p.value)) { previous = null; continue; }
            result += previous === null ? `M${x(p.row.end_step)},${y(p.value)} ` :
                stepwise ? `H${x(p.row.end_step)} V${y(p.value)} ` : `L${x(p.row.end_step)},${y(p.value)} `;
            previous = p;
        }
        return result;
    }
    const reference = guide ? `<path d="${path(guide, true)}" fill="none" stroke="#8893a3" stroke-width="2" stroke-dasharray="5 5"/>` : '';
    const marks = series.map(s => {
        const line = s.connect ? `<path d="${path(s.points)}" fill="none" stroke="${s.color}" stroke-width="1.7" opacity=".65"/>` : '';
        return line + s.points.filter(p => Number.isFinite(p.value)).map(p => {
            const attrs = `class="analysis-point" data-query="${esc(p.row.query_id)}" tabindex="0" role="button" aria-label="${esc(pointTitle(p.row, s.name + ': ' + p.value))}"`;
            const titleTag = `<title>${esc(pointTitle(p.row, s.name + ': ' + (s.probability ? p.value.toFixed(4) : p.value)))}</title>`;
            const failure = p.failed ? `<path d="M${x(p.row.end_step)-5},${y(p.value)-5} l10,10 m0,-10 l-10,10" stroke="#c64038" stroke-width="2" pointer-events="none"/>` : '';
            return (s.square ? `<rect ${attrs} x="${x(p.row.end_step)-6}" y="${y(p.value)-6}" width="12" height="12" fill="transparent" stroke="${s.color}" stroke-width="2">${titleTag}</rect>` :
                `<circle ${attrs} cx="${x(p.row.end_step)}" cy="${y(p.value)}" r="${s.probability ? 3 : 4.5}" fill="${s.color}" fill-opacity=".85">${titleTag}</circle>`) + failure;
        }).join('');
    }).join('');
    const gaps = statusRows.filter(r => !hasChoice(r)).map(r => {
        const color = r.status === 'pending' ? '#aab4c2' : '#c64038';
        return `<rect class="analysis-point" data-query="${esc(r.query_id)}" tabindex="0" role="button" aria-label="${esc(pointTitle(r, statuses[r.status]))}" x="${x(r.end_step)-3}" y="${H-54}" width="6" height="8" fill="${color}"><title>${esc(pointTitle(r, statuses[r.status]))}</title></rect>`;
    }).join('');
    return `<section class="panel"><h3>${esc(title)}</h3>${controls}<div class="analysis-legend" aria-label="回答系列の凡例">${analysisLegend(series)}</div><div class="annotation-legend"><span><i class="legend-annotation" aria-hidden="true"></i>薄い赤の縦線：注釈の発生step</span>${guide ? '<span><i class="legend-reference" aria-hidden="true"></i>灰破線：入力窓内に含まれる注釈の機体集合</span>' : ''}</div><div class="chart-scroll"><svg class="analysis-chart" viewBox="0 0 ${W} ${H}" role="group" aria-label="${esc(title)}">${grid}${annotationLines}${reference}${marks}${gaps}${axis}<text x="${W/2}" y="${H-3}" text-anchor="middle" font-size="12">入力窓の右端step（点を選ぶと入力・回答を表示）</text></svg></div>${statusRows.some(r => !hasChoice(r)) ? '<small class="muted">下段の灰印＝未実行、赤印＝エラー・不確定／4択なし。回答としては描画しません。</small>' : ''}</section>`;
}

function categoricalPlot(rows, api) {
    return analysisPlot({title: `${apiLabels[api]} · 回答の時系列`, rows, maxY: 3,
        ticks: choiceOrder.map((k, i) => ({value: i, label: choiceLabels[k]})),
        guide: rows.map(row => ({row, value: choiceOrder.indexOf(row.expected_aircraft_choice)})),
        statusRows: rows,
        series: choiceOrder.map(k => ({name: choiceLabels[k], color: choiceColors[k],
            points: rows.map(row => ({row, value: hasChoice(row) && row.aircraft_choice === k ? choiceOrder.indexOf(k) : null}))}))});
}

function differencePlots(responses, decisions) {
    const ticks = [0, 1, 2].map(value => ({value, label: `${value}機体`}));
    const referenceSeries = [
        {name: '青い丸：Responses と注釈の不一致数', color: '#1769aa', rows: responses},
        {name: '橙色の四角：Decisions と注釈の不一致数', color: '#cf6b15', rows: decisions, square: true}
    ].map(s => ({...s, points: s.rows.map(row => ({row, value: hasChoice(row) ? aircraftDistance(row.aircraft_choice, row.expected_aircraft_choice) : null}))}));
    const all = [...responses, ...decisions];
    const bySample = new Map(responses.map(row => [row.sample_id, row]));
    const paired = decisions.map(row => {
        const other = bySample.get(row.sample_id);
        return {row, value: hasChoice(row) && hasChoice(other) ? aircraftDistance(row.aircraft_choice, other.aircraft_choice) : null};
    });
    return analysisPlot({title: '注釈との不一致数 · 0＝一致 / 1＝片方 / 2＝両方', rows: all, ticks, maxY: 2, series: referenceSeries}) +
        analysisPlot({title: 'Decisions と Responses の不一致数 · 同じ窓・同じ反復', rows: all, ticks, maxY: 2,
            series: [{name: 'API間の差分（0でも正解とは限らない）', color: '#8b3db3', points: paired}]});
}

function probabilityPlot(rows) {
    return analysisPlot({title: 'Decisions · 4択の確率', rows, maxY: 1,
        ticks: [0, .25, .5, .75, 1].map(value => ({value, label: `${Math.round(value*100)}%`})),
        statusRows: rows,
        series: choiceOrder.map(k => ({name: choiceLabels[k], color: choiceColors[k], connect: true, probability: true,
            points: rows.map(row => ({row, value: hasChoice(row) ? (row.choice_probabilities?.[k] ?? null) : null}))}))});
}

function latencyPlot(responses, decisions) {
    const rows = [...responses, ...decisions];
    const elapsed = row => row.status !== 'pending' && Number.isFinite(row.latency_s) ? row.latency_s : null;
    const values = rows.map(elapsed).filter(Number.isFinite);
    const maxY = Math.max(1, Math.ceil(Math.max(0, ...values) / 10) * 10);
    const series = [
        {name: '青い丸：Responses（秒）', color: '#1769aa', rows: responses},
        {name: '橙色の四角：Decisions（秒）', color: '#cf6b15', rows: decisions, square: true}
    ].map(s => ({...s, connect: true, points: s.rows.map(row => ({row,
        value: elapsed(row), failed: row.status !== 'ok'}))}));
    const tickValues = latencyScale === 'log'
        ? [0, ...Array.from({length: Math.ceil(Math.log10(maxY)) + 1}, (_, i) => [1, 3].map(n => n * 10**i)).flat()].filter(v => v <= maxY)
        : [0, .25, .5, .75, 1].map(f => f * maxY);
    const controls = `<label>所要時間の縦軸<select class="latency-scale"><option value="log" ${latencyScale === 'log' ? 'selected' : ''}>対数：短い応答の差を見やすく</option><option value="linear" ${latencyScale === 'linear' ? 'selected' : ''}>線形：0秒から全体を比較</option></select></label><p class="muted">${latencyScale === 'log' ? '縦軸は log(1＋秒) による対数表示。目盛りは実際の秒数で、等間隔ではありません。外れ値も省略せず表示します。' : '縦軸は秒数に比例する通常のスケールです。'}</p>`;
    return analysisPlot({title: 'API所要時間 · 送信開始から応答／エラーまで（秒）', rows, series, maxY,
        scale: latencyScale, controls,
        ticks: tickValues.map(value => ({value, label: `${num(value,1)} 秒`}))}) +
        '<p class="muted">同じ反復の両APIを共通の秒軸で比較。赤い×を重ねた点はAPIエラー・形式エラー等です。タイムアウトの時間は打ち切りまでの待ち時間であり、正常な応答時間ではありません。未実行・時間未記録は欠測です。</p>';
}

function confusionTable(rows, keys, expectedKey, predictedKey, title, axisLabel = '注釈 ↓ ／ 回答 →') {
    const matrix = keys.map(() => keys.map(() => 0));
    for (const row of rows) {
        const i = keys.indexOf(row[expectedKey]), j = keys.indexOf(row[predictedKey]);
        if (i >= 0 && j >= 0) matrix[i][j]++;
    }
    const labels = k => choiceLabels[k] || k;
    return `<div><h3>${esc(title)}</h3><table class="matrix"><thead><tr><th>${esc(axisLabel)}</th>${keys.map(k => `<th>${esc(labels(k))}</th>`).join('')}<th>計</th></tr></thead><tbody>${matrix.map((cells, i) => {
        const total = cells.reduce((a,b) => a+b, 0);
        return `<tr><th>${esc(labels(keys[i]))}</th>${cells.map((count, j) => `<td style="background:rgba(${i===j ? '33,139,88' : '192,64,56'},${total ? .08+.4*count/total : 0})">${count}<small>${total ? (100*count/total).toFixed(1)+'%' : '—'}</small></td>`).join('')}<td>${total}</td></tr>`;
    }).join('')}</tbody></table><p class="matrix-note">件数と行内割合。分母0は「—」。反復をまとめる場合も各応答を1件と数えます。</p></div>`;
}

function confusionPanel(rows, api) {
    const all = rows.filter(r => r.condition === api), valid = all.filter(r => r.status === 'ok');
    if (!all.length) return '';
    const four = valid.filter(hasChoice), pending = all.filter(r => r.status === 'pending').length;
    const errors = all.length - valid.length - pending;
    const positive = valid.filter(r => r.expected_decision === 'YES');
    const tp = positive.filter(r => r.decision === 'YES').length;
    const negative = valid.filter(r => r.expected_decision === 'NO');
    const fp = negative.filter(r => r.decision === 'YES').length;
    const fraction = (n,d) => d ? `${n}/${d} (${(100*n/d).toFixed(1)}%)` : '—';
    return `<section class="panel"><h2>${apiLabels[api]} · 窓ごとの混同行列</h2><p>予定 ${all.length} / 有効 ${valid.length} / エラー・不確定 ${errors} / 未実行 ${pending}</p><p>注釈ありをYES：${fraction(tp,positive.length)}　注釈なしをYES：${fraction(fp,negative.length)}</p><div class="matrix-wrap">${confusionTable(valid, ['NO','YES'], 'expected_decision', 'decision', '注釈なし／あり × NO／YES')}${confusionTable(four, choiceOrder, 'expected_aircraft_choice', 'aircraft_choice', '機体集合の4択')}</div></section>`;
}

function apiConfusionPanel(rows) {
    const responses = rows.filter(r => r.condition === 'responses_diagnostic');
    const decisions = rows.filter(r => r.condition === 'decisions');
    if (!responses.length || !decisions.length) return '';
    const pairKey = r => `${r.repeat}:${r.sample_id}:${r.start_step}:${r.end_step}`;
    const lookup = new Map(decisions.map(r => [pairKey(r), r]));
    const pairs = responses.map(r => ({responses: r, decisions: lookup.get(pairKey(r))})).filter(p => p.decisions);
    const valid = pairs.filter(p => p.responses.status === 'ok' && p.decisions.status === 'ok');
    const binary = valid.map(p => ({responses: p.responses.decision, decisions: p.decisions.decision}));
    const aircraft = valid.filter(p => hasChoice(p.responses) && hasChoice(p.decisions))
        .map(p => ({responses: p.responses.aircraft_choice, decisions: p.decisions.aircraft_choice}));
    const agreement = values => values.length ? `${values.filter(p => p.responses === p.decisions).length}/${values.length} (${(100*values.filter(p => p.responses === p.decisions).length/values.length).toFixed(1)}%)` : '—';
    const axis = 'Responses ↓ ／ Decisions →';
    return `<section class="panel" id="api-confusion"><h2>Responses × Decisions · API間の混同行列</h2><p>同じ入力窓・同じ反復の予定ペア ${pairs.length} / 両方有効 ${valid.length} / エラー・未実行等による除外 ${pairs.length-valid.length}</p><p>YES/NOの一致率：${agreement(binary)}　機体4択の一致率：${agreement(aircraft)}</p><p class="muted">行がResponses、列がDecisionsです。対角線は両APIの一致を示し、注釈に対する正解を意味しません。件数とResponsesの各回答を分母にした行内割合を表示します。</p><div class="matrix-wrap">${confusionTable(binary, ['NO','YES'], 'responses', 'decisions', 'API間のYES/NO', axis)}${confusionTable(aircraft, choiceOrder, 'responses', 'decisions', 'API間の機体4択', axis)}</div></section>`;
}

function openAnalysisQuery(qid) {
    // Clear detail filters so an error or a point from either API always remains reachable.
    for (const id of ['condition','expected','result','repeat','aircraft-filter','search']) $(id).value = '';
    view = 'detail'; apply();
    index = filtered.findIndex(row => row.query_id === qid);
    if (index < 0) index = 0;
    setView('detail');
    window.scrollTo(0,0);
}

function renderAnalysis() {
    const repeats = [...new Set(run.rows.map(r => r.repeat))].sort((a,b) => a-b);
    if (analysisRepeat !== 'all' && !repeats.includes(Number(analysisRepeat))) analysisRepeat = String(repeats[0]);
    const selectedRepeats = analysisRepeat === 'all' ? repeats : [Number(analysisRepeat)];
    const rows = run.rows.filter(r => selectedRepeats.includes(r.repeat));
    let html = `<section class="panel notice"><h2>時系列と判定の比較</h2><div class="analysis-controls"><label>解析対象の反復<select id="analysis-repeat">${repeats.map(n => `<option value="${n}">r${String(n).padStart(2,'0')}</option>`).join('')}<option value="all">全反復（時系列は反復別）</option></select></label><span>このタブは上部の詳細フィルターを使わず、選んだ反復の全窓を表示します。</span></div><p>No＝どちらもなし / A2＝A0200 / A1＝A0100 / Both＝両方。縦軸はカテゴリで、重要度の順ではありません。横軸は回答が対象とした窓の右端で、モデルが指摘した発生stepとは異なります。</p><p>差分は機体ごとのYES/NOの不一致数です。A1↔A2、No↔Bothは2、A1↔Bothは1です。欠測は点を打たず、確率の線も欠測位置で切ります。重なった点は注釈差分の青丸・橙の枠で区別します。</p><p class="muted">混同行列は窓ごとの評価です。同じ注釈が複数窓に現れるため、独立したイベント数や初回検出率ではありません。未注釈は暫定負例です。3反復は同じ試合での回答のばらつきを表します。</p></section>`;
    for (const repeat of selectedRepeats) {
        const ordered = rows.filter(r => r.repeat === repeat).sort((a,b) => a.end_step-b.end_step);
        const decisions = ordered.filter(r => r.condition === 'decisions');
        const responses = ordered.filter(r => r.condition === 'responses_diagnostic');
        const validPairs = decisions.filter(d => hasChoice(d) && responses.some(r => r.sample_id === d.sample_id && hasChoice(r))).length;
        html += `<h2>r${String(repeat).padStart(2,'0')} · API間比較できる窓 ${validPairs} / ${decisions.length}</h2>`;
        if (decisions.length) html += categoricalPlot(decisions, 'decisions');
        if (responses.length) html += categoricalPlot(responses, 'responses_diagnostic');
        html += differencePlots(responses, decisions);
        if (decisions.length) html += probabilityPlot(decisions);
        html += latencyPlot(responses, decisions);
    }
    html += confusionPanel(rows, 'decisions') + confusionPanel(rows, 'responses_diagnostic') + apiConfusionPanel(rows);
    $('content').innerHTML = html;
    $('analysis-repeat').value = analysisRepeat;
    $('analysis-repeat').onchange = e => {analysisRepeat = e.target.value; renderAnalysis()};
    document.querySelectorAll('.latency-scale').forEach(select => {
        select.onchange = e => {latencyScale = e.target.value; renderAnalysis()};
    });
    document.querySelectorAll('.analysis-point').forEach(point => {
        point.onclick = () => openAnalysisQuery(point.dataset.query);
        point.onkeydown = e => {if (e.key === 'Enter' || e.key === ' ') {e.preventDefault(); openAnalysisQuery(point.dataset.query)}};
    });
}
