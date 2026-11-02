// Run with: node --test experiments/18_desicion_LAG/tests/test_analysis.cjs
// No browser or network dependencies; rendering interactions are also checked in jsdom.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const context = vm.createContext({
    esc: value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),
    statuses: {pending: '未実行', api_error: 'APIエラー'}
});
vm.runInContext(readFileSync(path.join(__dirname, '../templates/analysis.js'), 'utf8'), context);
const evaluate = source => vm.runInContext(source, context);

function row(end, choice, status = 'ok') {
    return {end_step: end, start_step: end-39, query_id: `r01_end_${end}_decisions`,
        repeat: 1, condition: 'decisions', status, aircraft_choice: choice,
        expected_aircraft_choice: 'A1', expected_decision: 'YES',
        decision: choice === 'NONE' ? 'NO' : 'YES',
        choice_probabilities: status === 'ok' ? {NONE: .1, A2: .2, A1: .3, BOTH: .4} : null};
}

test('distance counts aircraft disagreements, not distance on the plotted category axis', () => {
    assert.equal(evaluate("aircraftDistance('A1','A2')"), 2);
    assert.equal(evaluate("aircraftDistance('NONE','BOTH')"), 2);
    assert.equal(evaluate("aircraftDistance('A1','BOTH')"), 1);
    assert.equal(evaluate("aircraftDistance('NONE','A1')"), 1);
    assert.equal(evaluate("aircraftDistance('A2','A2')"), 0);
    assert.equal(evaluate("aircraftDistance(null,'NONE')"), null);
});

test('probability lines break at failed requests and missing values never become zero', () => {
    context.rows = [row(39, 'A1'), row(44, null, 'api_error'), row(49, 'A1')];
    const html = evaluate('probabilityPlot(rows)');
    const paths = [...html.matchAll(/<path d="([^"]*)"/g)].map(m => m[1]);
    assert.equal(paths.length, 4);
    for (const p of paths) {assert.equal((p.match(/M/g)||[]).length, 2); assert.ok(!p.includes('L'));}
    assert.ok(!html.includes('NaN'));
    assert.ok(html.includes('APIエラー'));
});

test('confusion matrices exclude errors and pending; BOTH to A1 remains a binary TP', () => {
    const partial = {...row(39,'A1'), expected_aircraft_choice: 'BOTH'};
    context.rows = [partial, row(44,null,'api_error'), row(49,null,'pending')];
    const html = evaluate("confusionPanel(rows,'decisions')");
    assert.ok(html.includes('予定 3 / 有効 1 / エラー・不確定 1 / 未実行 1'));
    assert.ok(html.includes('注釈ありをYES：1/1 (100.0%)'));
    const matrices = [...html.matchAll(/<tbody>(.*?)<\/tbody>/gs)].map(m=>m[1]);
    // Second matrix: fourth expected row (Both), third prediction column (A1).
    const rows = [...matrices[1].matchAll(/<tr>(.*?)<\/tr>/gs)].map(m=>m[1]);
    const cells = [...rows[3].matchAll(/<td[^>]*>(\d+)/g)].map(m=>Number(m[1]));
    assert.deepEqual(cells, [0,0,1,0,1]);
});

test('Responses raw text is retained even when output validation fails', () => {
    context.record = {status: 'invalid_output', raw: {output: [{type: 'message', content: [{type:'output_text',text:'{"decision":"YES","events":[]}' }]}]}};
    assert.equal(evaluate('responseText(record)'), '{"decision":"YES","events":[]}');
});
