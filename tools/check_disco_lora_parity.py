"""Compare the host distilled backend to the handoff's standalone tdm source."""
import argparse
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from safetensors.torch import load_file

from vibewarp.config import DISCO_DISTILLED_PRESET
from vibewarp.config_io import config_from_dict
from vibewarp.core.disco import tensor_image
from vibewarp.core.disco_lora import (AdapterLayer, DistilledDiscoBackend,
                                     adapter_config, resolve_adapter, sha256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('.cache/disco-lora-parity'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    adapter = resolve_adapter(args.adapter)
    cfg = adapter_config(adapter)
    sys.path.insert(0, str(args.source_root.resolve()))
    from tdm.core import load_model, inject_adapters, load_adapter_state, rollout
    from tdm.evaluation import autocast_context
    seeds = [12345,12346]
    reference = {}
    model, schedule = load_model(args.teacher,'cuda')
    inject_adapters(model,cfg['rank'],cfg['alpha'],names=('student',))
    load_adapter_state(model,load_file(str(adapter)),'student')
    model.eval().requires_grad_(False).to('cuda')
    print('Standalone reference loaded',flush=True)
    for seed in seeds:
        for active in ('student',None):
            noise = torch.randn((1,3,512,512),device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed))
            with torch.inference_mode(),autocast_context('fp16','cuda'):
                result,_ = rollout(model,schedule,noise,cfg['grid'],adapter=active)
            reference[(seed,active)] = result.cpu()
            tensor_image(result).save(args.output/f'reference_{seed}_{active}.png')
            print(f'Reference seed={seed}, adapter={active}',flush=True)
    del model,schedule,result,noise
    gc.collect()
    torch.cuda.empty_cache()
    config = config_from_dict(DISCO_DISTILLED_PRESET)
    config.disco.checkpoint_path = str(args.teacher.resolve())
    config.disco.few_step_lora_path = str(adapter)
    config.disco.steps = len(cfg['grid'])-1
    host = DistilledDiscoBackend(config)
    assert all(p.dtype == torch.float32 for p in host.model.parameters())
    report = {'adapter':str(adapter),'adapter_sha256':sha256(adapter),
        'reference_core_sha256':sha256(args.source_root/'tdm'/'core.py'),
        'torch':torch.__version__,'device':torch.cuda.get_device_name(),
        'grid':cfg['grid'],'adapter_layers':sum(isinstance(m,AdapterLayer) for m in host.model.modules()),
        'stored_parameter_dtype':'float32','samples':[]}
    calls,head_dtypes = [],[]
    active_name = [None]
    def capture(module,inputs):
        calls.append(int(inputs[1][0]))
        assert all(m.active == active_name[0] for m in module.modules() if isinstance(m,AdapterLayer))
    hook = host.model.register_forward_pre_hook(capture)
    head_hook = host.model.out.register_forward_hook(lambda module,inputs,output:head_dtypes.append(str(output.dtype)))
    for seed in seeds:
        for active in ('student',None):
            calls.clear()
            active_name[0] = active
            result = host.sample_tensor(seed,adapter=active).cpu()
            expected = reference[(seed,active)]
            error = result-expected
            assert calls == cfg['grid'][:-1],calls
            pixels = tensor_image(result)
            pixels.save(args.output/f'host_{seed}_{active}.png')
            pixel_equal = pixels.tobytes() == tensor_image(expected).tobytes()
            sample = {'seed':seed,'adapter':active,'timesteps':list(calls),
                'max_abs':float(error.abs().max()),'mae':float(error.abs().mean()),
                'tensor_equal':bool(torch.equal(result,expected)),'pixel_equal':pixel_equal}
            report['samples'].append(sample)
            print(json.dumps(sample),flush=True)
    hook.remove()
    head_hook.remove()
    assert set(head_dtypes) == {'torch.float32'},head_dtypes
    report['head_dtype'] = 'float32'
    report['adapter_effect_max_abs'] = float((reference[(seeds[0],'student')]-reference[(seeds[0],None)]).abs().max())
    assert report['adapter_effect_max_abs'] > 0
    report['all_tensor_equal'] = all(s['tensor_equal'] for s in report['samples'])
    report['all_pixel_equal'] = all(s['pixel_equal'] for s in report['samples'])
    (args.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if not report['all_tensor_equal']:
        raise RuntimeError('Strict tensor parity failed; see report before enabling this adapter')
    # Exercise the public render/preview route independently of the raw tensor check.
    from PIL import Image
    image = host.render(config,Image.new('RGB',(512,512)),'','',seeds[0],preview_path=args.output/'preview.png')
    assert image.tobytes() == tensor_image(reference[(seeds[0],'student')]).tobytes()
    image.save(args.output/'render.png')
    print('Strict reference parity and public render passed',flush=True)


if __name__ == '__main__':
    main()
