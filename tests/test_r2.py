"""CPU-only checks for public R2 routes and the fixed Mean-only interface."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
torch.set_num_threads(1)
from ha_paano.r2_configuration import expanded_jobs, validate_request, read_frozen, plan_r2
from ha_paano.r2_routes import ROUTES, CONDITIONS, clean_environment
from ha_paano.r2_controls import install_mean_only, rng_hash, state_hash
from ha_paano._reference.model_FINAL_HA_PaAno_CANONICAL import PatchEncoder
from ha_paano._reference import cross_branch_model
from ha_paano.configuration import frozen_environment

class R2Tests(unittest.TestCase):
    def test_01_manifest_protocol_and_scale(self):
        for track,split,count in [('U','tuning',48),('M','tuning',20),('M','evaluation',180)]:
            m=read_frozen(f'r2_{split}_{track}.json')
            self.assertEqual(len(m['members']),count)
            self.assertEqual(len({x['series'] for x in m['members']}),count)
            self.assertEqual([x['series'] for x in m['members']],sorted(x['series'] for x in m['members']))
        p=read_frozen('r2_protocol.json')
        self.assertEqual(p['num_iters'],100);self.assertEqual(p['batch_size'],512)
        self.assertFalse(p['scale_enabled']);self.assertFalse(p['shift_enabled'])
        jobs=expanded_jobs();t=[j for j in jobs if j['split']=='tuning'];e=[j for j in jobs if j['split']=='evaluation']
        self.assertEqual((len(t),sum(j['fits'] for j in t)),(51,1692))
        self.assertEqual((len(e),sum(j['fits'] for j in e)),(3,540))
        self.assertEqual(len({tuple(j.values()) for j in jobs}),54)

    def test_02_illegal_requests(self):
        for args in [('U','C',2000,'tuning',True),('M','JLC',2000,'tuning',True),('M','C',2003,'tuning',False),('M','C',2000,'evaluation',False),('U','B',2000,'evaluation',False)]:
            with self.assertRaises(ValueError): validate_request(*args)

    def test_03_clean_env(self):
        env,_=clean_environment({'PAANO_INTERNAL_VARIANT':'no_gate','PAANO_CROSS_CONDITION':'bad','KEEP':'x'},'M','C')
        self.assertNotIn('PAANO_INTERNAL_VARIANT',env);self.assertNotIn('PAANO_CROSS_CONDITION',env)
        self.assertEqual(env['PAANO_URPA_TRANSFORM_VARIANT'],'jitter_only');self.assertEqual(env['KEEP'],'x')

    def test_04_mean_only_exact_descriptor_and_guards(self):
        torch.manual_seed(2000)
        model=PatchEncoder(in_channels=3,use_revin=True)
        x=torch.arange(4*3*96,dtype=torch.float32).reshape(4,3,96)/100
        raw=[];masked=[]
        first=model.channel_context.context_mlp.register_forward_pre_hook(lambda m,i:raw.append(i[0].detach().clone()))
        before=rng_hash();state=state_hash(model);schema=list(model.state_dict())
        install_mean_only(model)
        self.assertEqual(before,rng_hash());self.assertEqual(state,state_hash(model));self.assertEqual(schema,list(model.state_dict()))
        second=model.channel_context.context_mlp.register_forward_pre_hook(lambda m,i:masked.append(i[0].detach().clone()))
        for training in (True,False):
            model.train(training);model.embedding(x)
            self.assertTrue(torch.equal(masked[-1],raw[-1]*raw[-1].new_tensor([1,0,0,0,1,0,0,0])))
        first.remove();second.remove()
        with self.assertRaises(RuntimeError): install_mean_only(model)
        with frozen_environment('U'):
            with self.assertRaises(ValueError): install_mean_only(cross_branch_model.PatchEncoder(in_channels=1,use_revin=True))
        # Removing a non-parameter hook restores the unmodified full-C behavior.
        model.eval()
        saved=model.state_dict();model._r2_mean_only_handle.remove();del model._r2_mean_only_handle
        reference=PatchEncoder(in_channels=3,use_revin=True).eval();reference.load_state_dict(saved)
        with torch.no_grad(): self.assertTrue(torch.equal(model.embedding(x),reference.embedding(x)))

    def test_05_hash_rejects_wrong_data_without_output(self):
        member=read_frozen('r2_tuning_U.json')['members'][0]
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'input';folder.mkdir();(folder/member['series']).write_text('x,label\n1,0\n')
            output=Path(temp)/'output'
            one={'members':[member],'expected_series':1}
            with patch('ha_paano.r2_configuration.read_frozen',side_effect=[read_frozen('r2_protocol.json'),one]):
                with self.assertRaisesRegex(ValueError,'SHA256/size'): plan_r2('U','B',2000,'tuning',folder,output)
            self.assertFalse(output.exists())

    def test_06_cli_dry_does_not_launch(self):
        spec=importlib.util.spec_from_file_location('r2_cli',ROOT/'scripts/run_r2_experiment.py')
        cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        import contextlib,io
        args=['--track','U','--condition','B','--seed','2000','--split','tuning','--data-dir','unused','--output-dir','unused']
        with patch.object(cli,'plan_r2',return_value={'training_started':False}),patch.object(cli.subprocess,'run') as run:
            with contextlib.redirect_stdout(io.StringIO()): self.assertEqual(cli.main(args),0)
            run.assert_not_called()
        fake={'track':'U','condition':'B','execution':None}
        with patch.object(cli,'plan_r2',return_value=fake),patch.object(cli.subprocess,'run',return_value=subprocess.CompletedProcess([],7)) as run:
            with contextlib.redirect_stdout(io.StringIO()): self.assertEqual(cli.main(args+['--execute']),7)
            self.assertEqual(run.call_args.args[0][0],sys.executable)
            self.assertEqual(run.call_args.args[0][-1],'--_worker')
            self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'-1')
        plan={'track':'M','condition':'C','execution':read_frozen('r2_eva_execution.json'),'seed':2001}
        env=cli.child_environment(plan,'cpu')
        self.assertEqual(env['OMP_NUM_THREADS'],'1');self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'-1')
        self.assertNotIn('LOKY_MAX_CPU_COUNT',env)

    def test_07_all_routes_and_mean_only_in_fresh_cpu_processes(self):
        cases=[(t,c,False) for t in ('U','M') for c in CONDITIONS]+[('M','C',True)]
        def run(case):
            t,c,mean=case
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
            args=[sys.executable,'-B',str(ROOT/'tests/r2_probe.py'),'--track',t,'--condition',c]
            if mean: args+=['--mean-only']
            p=subprocess.run(args,capture_output=True,text=True,encoding='utf-8',env=env,timeout=120)
            if p.returncode: raise AssertionError(f'{case}: {p.stderr[-4000:]}')
            result=json.loads(p.stdout)
            assert result['optimizer_steps']==6 and not result['cuda_initialized']
            return result
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(run,cases))
        self.assertEqual(len(results),17)

if __name__=='__main__': unittest.main()
