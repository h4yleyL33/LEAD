const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(process.argv[2] || path.join(__dirname, '../backend/app/static/index.html'), 'utf8');
const script = html.split('<script>')[1].split('</script>')[0];
const ctx = vm.createContext({document: {readyState: 'loading', querySelector() { return null; }, addEventListener() {}}, window: {addEventListener() {}}});
// Compile and evaluate declarations: duplicate consts and missing view functions fail here.
vm.runInContext(script, ctx);
assert.equal(vm.runInContext('ALIASES[normCol("Meter Reading")]', ctx), 'meter_reading');
assert.equal(vm.runInContext('Object.keys(VIEWS).length', ctx), 7);
for (const name of ['vOverview', 'vMonitor', 'vLive', 'vDatasets', 'vEvents', 'vSettings', 'vConnection']) {
  assert.match(vm.runInContext(`${name}()`, ctx), /<(section|div)/);
}
const records = [['a,b', 'a"b', 'line\nbreak', '']];
ctx.records = records;
assert.equal(JSON.stringify(vm.runInContext('parseCSV(toCSV(["one","two","three","four"], records)).slice(1)', ctx)), JSON.stringify(records));
assert.equal(vm.runInContext('busy("load", true); busy("load")', ctx), true);
assert.equal(vm.runInContext('busy("load", false); busy("load")', ctx), false);
vm.runInContext('S.refBuilding=32; S.threshold=0.5; S.refSeries={points:[{t:"2026-11-18T00:00",v:12,score:0.8}]};', ctx);
assert.equal(vm.runInContext('eventsRows()[0].alarm', ctx), 'ANOMALY');
console.log('Dashboard declarations, seven views, aliases, CSV round-trip, busy state: OK');


ctx.document.querySelector = s => s === '#live-asset' ? {value:'0'} : null;
ctx.setInterval = () => 1;
ctx.clearInterval = () => {};
vm.runInContext('pollLive = () => {}; startLive();', ctx);
assert.equal(vm.runInContext('S.liveId', ctx), 0);
