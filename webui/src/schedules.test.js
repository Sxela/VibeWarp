import {describe,expect,it} from 'vitest';
import {canonicalSchedule,normalizeSchedules} from './schedules.js';

const number={type:'any',schedule_domain:'frame'};
const schema={properties:{
  text_prompts:{schedule_kind:'text',schedule_domain:'frame'},
  disco:{type:'dataclass',properties:{
    init_scale:{type:'number',tier:'hidden'},image_scale:{type:'number',tier:'hidden'},
    init_scale_schedule:{...number,schedule_fallback:'init_scale',schedule_interpolation:'hold'},
    image_scale_schedule:{...number,schedule_fallback:'image_scale',schedule_interpolation:'hold'},
    cut_overview:{schedule_domain:'inference_progress',schedule_value:'cut_count'},
  }},
  diffusion:{type:'dataclass',properties:{cfg_scale_schedule:number,sampler_scale_schedule:{schedule_domain:'sampling_step'}}},
}};
describe('settings schedule normalization',()=>{
  it.each([0,1000])('loads scalar-only frame guidance as frame 0 including zero: %s',scale=>{
    const input={disco:{init_scale:scale,image_scale:0,init_scale_schedule:null}};
    expect(normalizeSchedules(input,schema).disco).toEqual({...input.disco,init_scale_schedule:{0:scale},image_scale_schedule:{0:0}});
    expect(input.disco.init_scale_schedule).toBeNull();
  });
  it('keeps explicit schedules authoritative over scalar fallbacks',()=>{
    const input={disco:{init_scale:1000,image_scale:0,init_scale_schedule:{0:0},image_scale_schedule:{0:0,1:50000}}};
    expect(normalizeSchedules(input,schema)).toEqual(input);
  });
  it('retains per-step ramps, fractional progress keys and empty prompt dictionaries',()=>{
    const input={text_prompts:{},diffusion:{cfg_scale_schedule:[[3,7,3]],sampler_scale_schedule:{0:50,15:100}},disco:{cut_overview:{0:4,33.25:2,100:0}}};
    expect(normalizeSchedules(input,schema)).toEqual(input);
  });
  it('normalizes flat lists and prompt lists without losing keys or strings',()=>{
    expect(normalizeSchedules({text_prompts:{0:['a','b'],24:'scene'},diffusion:{cfg_scale_schedule:[7,9]}},schema))
      .toEqual({text_prompts:{0:'a | b',24:'scene'},diffusion:{cfg_scale_schedule:{0:7,1:9}}});
  });
  it('does not invent overrides for optional absent schedules or alter unknown data',()=>{
    const input={diffusion:{cfg_scale_schedule:null},extra:{schedule:[1,2]}};
    expect(normalizeSchedules(input,schema)).toEqual(input);
  });
  it('does not activate optional schedules over visible mode-specific scalars',()=>{
    const input={disco:{sampler:'distilled',distilled_guided:false,steps:4,steps_schedule:null}};
    const schema={properties:{disco:{type:'dataclass',properties:{steps:{type:'number',tier:'render'},steps_schedule:{...number,schedule_fallback:'steps'}}}}};
    expect(normalizeSchedules(input,schema)).toEqual(input);
  });
  it('validates each axis and value domain on paste',()=>{
    expect(()=>canonicalSchedule({'33.25':2})).toThrow(/whole/);
    expect(canonicalSchedule({'33.25':2},{axis:'progress',valueType:'cut_count'})).toEqual({'33.25':2});
    expect(()=>canonicalSchedule({101:2},{axis:'progress'})).toThrow(/between/);
    expect(()=>canonicalSchedule({0:1.5},{valueType:'cut_count'})).toThrow(/invalid/);
    expect(()=>canonicalSchedule({0:2},{valueType:'probability'})).toThrow(/invalid/);
  });
});
