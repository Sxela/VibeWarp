"""Sequential fixed-noise guidance corrections before one DDIM advance."""
import math
from types import MethodType


def skip_for_strength(steps, strength):
    # Frame schedules often encode an exact integer skip as a repeating float.
    return min(steps-1, int(steps*(1-strength)+1e-8))


def cut_batches_per_gradient(batches,repeats=1,divide=False):
    if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in (batches,repeats)):
        raise ValueError('cutn_batches and CLIP_GRAD_STEPS must be positive integers')
    if not divide:return [batches]*repeats
    if repeats>batches:raise ValueError('Split cuts requires cutn_batches >= CLIP_GRAD_STEPS')
    quotient,remainder=divmod(batches,repeats)
    return [quotient+(i<remainder) for i in range(repeats)]


def install_multi_guidance(diffusion,repeats=1,divide_cuts=False,cutn_batches=4):
    counts=cut_batches_per_gradient(cutn_batches,repeats,divide_cuts)
    base=getattr(diffusion,'_multi_guidance_original_step',diffusion.ddim_sample)
    diffusion.ddim_sample=base
    diffusion.multi_guidance_settings=dict(repeats=repeats,divide_cuts=bool(divide_cuts),cut_batches=counts)
    if repeats==1:return diffusion
    diffusion._multi_guidance_original_step=base
    def step(self,model,x,t,**kwargs):
        import torch
        kwargs.setdefault('model_kwargs', {})
        cond=kwargs.get('cond_fn')
        if cond is None:return base(model,x,t,**kwargs)
        eta=kwargs.get('eta',0.)
        if not math.isfinite(eta) or not 0<=eta<=1:raise ValueError('Repeated guidance requires DDIM eta in[0,1]')
        if not bool((t==t[0]).all()):raise ValueError('Repeated guidance requires one noise level per batch')
        index=int(t[0]);abar=float(self.alphas_cumprod[index]);previous=float(self.alphas_cumprod_prev[index])
        sigma=math.sqrt(1-abar)
        variance=eta**2*(1-previous)/(1-abar)*(1-abar/previous)
        gain=abs((math.sqrt(previous/abar)*sigma-math.sqrt(max(0.,1-previous-variance)))*sigma)
        propagation=math.sqrt(previous/abar)
        wrapped=self._wrap_model(cond) if hasattr(self,'_wrap_model') else cond
        pass_index=0
        def gradient(state):
            nonlocal pass_index
            count=counts[pass_index];pass_index+=1
            arguments=dict(kwargs.get('model_kwargs') or {})
            if divide_cuts:arguments['_clip_cutn_batches']=count
            timesteps=t if hasattr(self,'_wrap_model') else self._scale_timesteps(t)
            value=wrapped(state,timesteps,**arguments)
            weight=count/cutn_batches if divide_cuts else 1/repeats
            return value*weight
        work=x
        for _ in range(repeats-1):work=(work+gain/propagation*gradient(work)).detach()
        noise_rng={}
        def final_gradient(state,*args,**kw):
            value=gradient(state)
            noise_rng['cpu']=torch.get_rng_state()
            if state.is_cuda:
                noise_rng['cuda']=torch.cuda.get_rng_state(state.device)
            return value
        kwargs['cond_fn']=final_gradient
        cap=getattr(self,'guidance_update_cap',None)
        if cap is not None:
            self.guidance_update_cap=None
        try:
            result=base(model,work,t,**kwargs)
            if cap is not None:
                # Bound the complete repeated correction, including the first
                # passes. Replay only the final noise draw for the baseline.
                devices=[x.device.index] if x.is_cuda else []
                with torch.random.fork_rng(devices=devices), torch.no_grad():
                    torch.set_rng_state(noise_rng['cpu'])
                    if x.is_cuda:torch.cuda.set_rng_state(noise_rng['cuda'],x.device)
                    baseline=base(model,x,t,**{**kwargs,'cond_fn':None})
                result['sample']=cap(result['sample'],baseline['sample'],sigma)
        finally:
            if cap is not None:self.guidance_update_cap=cap
        if pass_index!=repeats:raise RuntimeError('Unexpected repeated-guidance call count')
        return result
    diffusion.ddim_sample=MethodType(step,diffusion)
    return diffusion


def validate_grid(grid, steps):
    if (len(grid) != steps+1 or grid[-1] != -1
            or any(type(t) is not int for t in grid)
            or not all(0 <= t < 1000 for t in grid[:-1])
            or any(a <= b for a,b in zip(grid,grid[1:]))):
        raise ValueError('Grid requires steps+1 strictly descending integer times in [0,999], ending -1')
    return grid
