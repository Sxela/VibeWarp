import {it,expect} from 'vitest';
import {sortedComfyModels,prefixLength} from './comfyModels.js';

it('pins a missing suggestion and sorts remaining names by lowercase prefix then alphabetically',()=>{
  const suggested='moss_model.safetensors';
  const models=['Zeta','moss_zzz','MOSS_model_alt','moss','alpha','moss_other','MOSS_model_aaa'];
  expect(sortedComfyModels(models,suggested)).toEqual([suggested,
    'MOSS_model_aaa','MOSS_model_alt','moss_other','moss_zzz','moss','alpha','Zeta']);
  expect(models[0]).toBe('Zeta');
  expect(prefixLength('QwEn3vl','qwen3VL_other')).toBe(7);
});

it('deduplicates the suggestion and preserves an imported missing selection',()=>{
  expect(sortedComfyModels(['preferred','other','preferred'],'preferred','custom')).toEqual(['preferred','other','custom']);
  expect(sortedComfyModels([], 'preferred','custom')).toEqual(['preferred','custom']);
});
