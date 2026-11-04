"""Synthetic CPU probe, never a benchmark. Optional frozen-source comparison.

Run from tests or provide --reference-root (a copied R2 source_snapshot),
--reference-routes and --reference-worker to compare against an isolated fixture.
No input signals, historical scores or checkpoints are bundled here.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['MPLBACKEND'] = 'Agg'
from pathlib import Path
import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import random
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
torch.set_num_threads(1)

def digest(value):
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            a=x.detach().cpu().contiguous().numpy()
            h.update(str((a.dtype, a.shape)).encode()); h.update(a.tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str((x.dtype, x.shape)).encode()); h.update(x.tobytes())
        elif isinstance(x, dict):
            for k,v in x.items(): h.update(str(k).encode()); visit(v)
        elif isinstance(x, (tuple,list)):
            for v in x: visit(v)
        else: h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()

def import_file(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track', choices=('U','M'), required=True)
    parser.add_argument('--condition', required=True)
    parser.add_argument('--mean-only', action='store_true')
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--reference-routes', type=Path)
    parser.add_argument('--reference-worker', type=Path)
    parser.add_argument('--reference-diagnostics', type=Path)
    args=parser.parse_args()
    if args.reference_root:
        base=args.reference_root
        for rel in ['', 'revision_experiments/cross_branch_full_pilot_3seed/scripts',
                    'revision_experiments/factorial_completion_1seed/scripts',
                    'revision_experiments/module_hyperparameter_sensitivity/scripts',
                    'revision_experiments/urpa_transform_only_pilot_3seed/scripts']:
            sys.path.insert(0,str(base/rel))
        routes=import_file('reference_routes', args.reference_routes)
        tree=ast.parse(args.reference_worker.read_text(encoding='utf-8-sig'))
        wanted={'seed_series','construct_training_input'}
        functions=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted], type_ignores=[])
        if len(functions.body)!=2:
            raise RuntimeError('Unexpected reference worker function interface')
        ns={'Path':Path}
        exec(compile(functions,str(args.reference_worker),'exec'),ns)
        make=lambda path,seed,device,model: ns['construct_training_input']({'canonical_path':str(path)},seed,device,model)
        from utils.utils import create_memory_bank
        from utils.evaluation import calculate_anomaly_scores, distribute_patch_scores_to_points
        controls=import_file('reference_diagnostics', args.reference_diagnostics)
    else:
        from ha_paano import r2_routes as routes
        from ha_paano.r2_runner import construct_training_input as make
        from ha_paano import r2_controls as controls
        from ha_paano._reference.utils.utils import create_memory_bank
        from ha_paano._reference.utils.evaluation import calculate_anomaly_scores, distribute_patch_scores_to_points
    model_module, trainer, _=routes.bind(args.track,args.condition)
    channels=1 if args.track=='U' else 3
    with tempfile.TemporaryDirectory() as temporary:
        path=Path(temporary)/'001_TOY_id_1_Test_tr_320_1st_400.csv'
        # Fixed arithmetic input, not a newly seeded real experiment.
        t=np.arange(512,dtype=np.float64)
        features=np.stack([np.sin(t/(11+c))+0.03*np.cos(t/3+c)+c*.4 for c in range(channels)],axis=1)
        labels=((t>=400)&(t<420)).astype(np.float64)
        np.savetxt(path,np.column_stack([features,labels]),delimiter=',',header=','.join([f'x{i}' for i in range(channels)]+['label']),comments='')
        ctx=make(path,2000,torch.device('cpu'),model_module)
        model=ctx['model']
        result={'initial_state':digest(model.state_dict()),'constructor_rng':digest(torch.get_rng_state()),
                'probe_indices':digest(ctx['probe_indexes']), 'patches':digest(ctx['patches']),
                'labels':digest(ctx['labels']),'window':int(ctx['window']),
                'parameter_count':sum(p.numel() for p in model.parameters()),
                'schema':[(k,list(v.shape),str(v.dtype)) for k,v in model.state_dict().items()]}
        route_condition=routes.ROUTES[(args.track,args.condition)][2]
        spec={"jitter":args.condition in ('J','JL','JC','JLC'),
              "lsdp":model.dual_pooling_head is not None if hasattr(model,'dual_pooling_head') else False,
              "cac":getattr(model,'channel_context',None) is not None}
        expected={'jitter':'J' in args.condition,'lsdp':'L' in args.condition,'cac':'C' in args.condition}
        assert spec==expected, (spec,expected)
        result['features']=spec
        result['route_condition']=route_condition
        before=controls.rng_hash();before_state=digest(model.state_dict())
        if args.mean_only:
            controls.install_mean_only(model)
            assert controls.rng_hash()==before and digest(model.state_dict())==before_state
        # Compare a real next-loader call as well as the pre-constructor probe.
        loader_batch,loader_indexes=next(iter(ctx['train_loader']))
        result['next_loader']=digest([loader_batch,loader_indexes])
        idx=torch.tensor([[100],[125],[150],[175]])
        batch=ctx['patches'][idx.flatten()].clone()
        model.eval()
        with torch.no_grad(): result['eval_forward']=digest(model.embedding(batch))
        model.train()
        phases=[]
        mask_checks=[]
        if args.mean_only:
            phase=['train']
            def inspect(module, inputs):
                descriptor=inputs[0].detach()
                mask_checks.append(bool(torch.equal(descriptor[:,[1,2,3,5,6,7]],torch.zeros_like(descriptor[:,[1,2,3,5,6,7]]))))
                phases.append(phase[0])
            inspection=model.channel_context.context_mlp.register_forward_pre_hook(inspect)
        gradients=[];updates=[];optimizer=[];losses=[]
        original_adam=torch.optim.AdamW
        class RecordedAdamW(original_adam):
            def step(self,*a,**kw):
                gradients.append(digest([p.grad for g in self.param_groups for p in g['params']]))
                value=super().step(*a,**kw)
                updates.append(digest(model.state_dict())); optimizer.append(digest(self.state_dict()))
                return value
        item=torch.Tensor.item
        def observed(tensor,*a,**kw):
            value=item(tensor,*a,**kw)
            if tensor.requires_grad and tensor.numel()==1: losses.append(float(value))
            return value
        torch.optim.AdamW=RecordedAdamW
        torch.Tensor.item=observed
        try:
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                trainer.train_model(model,[(batch,idx)],ctx['patches'],torch.device('cpu'),num_iter=6,pretext_step=96,lr=1e-4,see_loss=False)
        finally:
            torch.optim.AdamW=original_adam;torch.Tensor.item=item
        result.update({'gradients':gradients,'updates':updates,'optimizer_states':optimizer,
                       'losses':losses,'best_restored_state':digest(model.state_dict())})
        assert len(updates)==6 and all(np.isfinite(losses))
        if args.mean_only: phase[0]='memory'
        bank,_=create_memory_bank(model,[(ctx['patches'][:32],torch.arange(32).view(-1,1))],torch.device('cpu'),num_cores=.1)
        if args.mean_only: phase[0]='query'
        patch_scores=calculate_anomaly_scores(model,[(ctx['patches'][:12],torch.arange(12).view(-1,1))],bank,torch.device('cpu'),top_k=3)
        point_scores=distribute_patch_scores_to_points(patch_scores,patch_size=96,num_points=107)
        result.update({'memory_bank':digest(bank),'patch_scores':digest(np.array(patch_scores)),
                       'point_scores':digest(point_scores),'final_torch_rng':digest(torch.get_rng_state()),
                       'final_python_rng':digest(random.getstate()),'final_numpy_rng':digest(np.random.get_state()),
                       'optimizer_steps':len(updates),'cuda_initialized':torch.cuda.is_initialized()})
        if args.mean_only:
            inspection.remove()
            assert all(mask_checks) and set(phases)=={'train','memory','query'}
            result['mean_only_phases']=sorted(set(phases))
            result['mean_only_all_masked']=all(mask_checks)
        assert not torch.cuda.is_initialized()
        print(json.dumps(result,allow_nan=False))

if __name__=='__main__':
    main()
