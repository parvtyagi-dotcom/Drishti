/**
 * Checks for src/lib/alertPlayback.ts - when the forensic alert siren fires.
 *
 * The frontend has no test runner, and adding one for a single pure module is
 * not worth the dependency. Node strips the type annotations itself, so this
 * imports the module directly:
 *
 *     npm run check:alerts
 */
import { alertsReachedBy } from "../src/lib/alertPlayback.ts";

const ALERTS = [
  { key: "confirmed", timestampSec: 3.5 },
  { key: "intrusion", timestampSec: 8.0 },
  { key: "vehicle", timestampSec: 8.2 },
];

let passed = 0;
let failed = 0;

function check(name, got, want) {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  ok ? passed++ : failed++;
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${name}`);
  if (!ok) console.log(`          got  ${JSON.stringify(got)}\n          want ${JSON.stringify(want)}`);
}

const fired = (result) => result.due.map((alert) => alert.key);
const heard = (set) => [...set].sort();

console.log("\nordinary playback");
let sounded = new Set();
let result = alertsReachedBy(ALERTS, 0, 0.25, sounded);
sounded = result.sounded;
check("nothing sounds before the first alert", fired(result), []);

result = alertsReachedBy(ALERTS, 3.25, 3.5, sounded);
sounded = result.sounded;
check("sounds as playback reaches 3.5s", fired(result), ["confirmed"]);

result = alertsReachedBy(ALERTS, 3.5, 3.75, sounded);
sounded = result.sounded;
check("does not repeat on the following tick", fired(result), []);

result = alertsReachedBy(ALERTS, 7.9, 8.25, sounded);
sounded = result.sounded;
check("two alerts a fraction apart arrive together", fired(result), ["intrusion", "vehicle"]);

result = alertsReachedBy(ALERTS, 8.25, 8.5, sounded);
check("nothing is left to sound", fired(result), []);

console.log("\nseeking");
sounded = new Set();
result = alertsReachedBy(ALERTS, 0.2, 9.0, sounded);
sounded = result.sounded;
check("scrubbing past three alerts sounds none of them", fired(result), []);
check("...but counts all three as heard", heard(sounded), ["confirmed", "intrusion", "vehicle"]);

result = alertsReachedBy(ALERTS, 9.0, 1.0, sounded);
sounded = result.sounded;
check("seeking backwards sounds nothing immediately", fired(result), []);
check("...and re-arms the alerts ahead", heard(sounded), []);

result = alertsReachedBy(ALERTS, 3.4, 3.6, sounded);
check("replaying that stretch sounds it again", fired(result), ["confirmed"]);

console.log("\nrestarting");
sounded = new Set(["confirmed", "intrusion", "vehicle"]);
result = alertsReachedBy(ALERTS, 12.0, 0.0, sounded);
sounded = result.sounded;
check("playing again from the start re-arms everything", heard(sounded), []);
result = alertsReachedBy(ALERTS, 3.3, 3.55, sounded);
check("...and the first alert sounds on the replay", fired(result), ["confirmed"]);

console.log(`\n${passed} passed, ${failed} failed\n`);
process.exit(failed ? 1 : 0);
