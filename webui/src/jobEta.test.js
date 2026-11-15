import {expect,it} from 'vitest';
import {remainingEta,formatEta} from './jobEta.js';

it('counts down the outer render estimate and formats the supplied log example',()=>{
  const job={state:'running',eta_seconds:5646,eta_updated_at:'2026-11-15T11:12:31Z'};
  const remaining=remainingEta(job,Date.parse('2026-11-15T11:13:31Z'));
  expect(remaining).toBe(5586);expect(formatEta(5646)).toBe('1h 34m');
  expect(formatEta(56)).toBe('56s');expect(formatEta(123)).toBe('2m 3s');
  expect(remainingEta(job,Date.parse('2026-11-16T11:13:31Z'))).toBe(0);
});
it('hides unavailable, finished, cancelled and cleanup estimates',()=>{
  for(const job of [null,{state:'running'}, {state:'completed',eta_seconds:5},
    {state:'running',eta_seconds:5,cancel_requested:true},{state:'running',stage:'cleanup',eta_seconds:5}])expect(remainingEta(job)).toBeNull();
});
