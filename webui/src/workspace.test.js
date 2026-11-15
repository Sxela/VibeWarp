import {expect,it} from 'vitest';
import {migrateLegacyFrameLimit} from './workspace.js';
import {normalizeRawOpacities} from './workspace.js';
import {fieldGroup,fieldEditor,fieldValue,searchFields} from './workspace.js';

it('uses backend group/editor overrides while preserving the contextual custom editor',()=>{
  const entry={group:'ControlNet',schema:{editor:'controlnet',workflow_groups:{background:'ControlNet masks'},workflow_editors:{background:null}}};
  expect(fieldGroup(entry,'background')).toBe('ControlNet masks');
  expect(fieldEditor(entry,'background')).toBeNull();
  expect(fieldGroup(entry,'conditioning')).toBe('ControlNet');
  expect(fieldEditor(entry,'conditioning')).toBe('controlnet');
  expect(searchFields([{...entry,name:'qr_cn_mask_grayscale',tier:'advanced',section:'controlnet'}],'ControlNet masks')).toHaveLength(1);
});

it('shows the declared consistency weight fallback, including an explicit zero',()=>{
  const entry={section:'flow',name:'missed_consistency_schedule',schema:{schedule_fallback:'missed_consistency_weight'}};
  expect(fieldValue({flow:{missed_consistency_schedule:null,missed_consistency_weight:1}},entry)).toBe(1);
  expect(fieldValue({flow:{missed_consistency_schedule:null,missed_consistency_weight:0}},entry)).toBe(0);
  expect(fieldValue({flow:{missed_consistency_schedule:{0:.5},missed_consistency_weight:1}},entry)).toEqual({0:.5});
});

it('repairs early whole-percent raw opacity entries without changing valid fractions',()=>{
  const saved={mask:{foreground_opacity:50,background_opacity:75}};
  expect(normalizeRawOpacities(saved).mask).toEqual({foreground_opacity:0.5,background_opacity:0.75});
  expect(saved.mask.foreground_opacity).toBe(50);
  expect(normalizeRawOpacities({mask:{foreground_opacity:1,background_opacity:0.25}}).mask)
    .toEqual({foreground_opacity:1,background_opacity:0.25});
  expect(normalizeRawOpacities({mask:{foreground_opacity:1.1}}).mask.foreground_opacity).toBe(1.1);
});

it('drops stale browser caps and preserves an edited range',()=>{
  const saved={frame_range:[0,10],video:{max_frames:1,video_init_path:'clip.mp4'}};
  expect(migrateLegacyFrameLimit(saved)).toEqual({frame_range:[0,10],video:{video_init_path:'clip.mp4'}});
  expect(saved.video.max_frames).toBe(1);
});

it('converts an old preview cap once and leaves no second limiter',()=>{
  expect(migrateLegacyFrameLimit({frame_range:[0,0],video:{max_frames:1}})).toEqual({frame_range:[0],video:{}});
  expect(migrateLegacyFrameLimit({video:{max_frames:5}})).toEqual({frame_range:[0,4],video:{}});
});
