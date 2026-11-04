"""Small synthetic CPU-only checks; no research data and no full experiment."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['MPLBACKEND'] = 'Agg'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import csv
import json
import tempfile
import unittest
import subprocess
from unittest.mock import patch
import numpy as np
import torch
from ha_paano.configuration import frozen_environment, inspect_input, read_config
from ha_paano.runner import plan_run
from ha_paano._reference import cross_branch_model, cross_branch_train, cac_variants
from ha_paano._reference import train_FINAL_M_branch, pipeline
from ha_paano._reference.utils.data_preprocess import load_and_split_data, preprocess_to_patches, PatchCreator
from ha_paano._reference.utils.evaluation import distribute_patch_scores_to_points
from ha_paano._reference.utils.metrics import get_metrics
from ha_paano._reference.utils.result_saver import save_one_file_result

torch.set_num_threads(1)
TEST_TMP = Path(__file__).resolve().parents[1]/'.release_audit'
TEST_TMP.mkdir(exist_ok=True)

class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2000)
        np.random.seed(2000)

    def test_configs_and_file_manifests(self):
        for track, count in [('U',350),('M',180)]:
            self.assertEqual(read_config(track)['num_iters'],100)
            manifest = json.loads((Path(__file__).resolve().parents[1]/f'configs/evaluation_{track}.json').read_text())
            self.assertEqual(len(manifest['series']),count)
            self.assertEqual(len(set(manifest['series'])),count)

    def test_U_forward_raw_context_and_rng(self):
        with frozen_environment('U'):
            model = cross_branch_model.PatchEncoder(in_channels=1,use_revin=True).eval()
            self.assertEqual(sum(p.numel() for p in model.parameters()),376258)
            x = torch.randn(4,1,96)
            state = torch.get_rng_state().clone()
            with torch.no_grad():
                h = model.embedding(x)
            self.assertEqual(tuple(h.shape),(4,64))
            self.assertTrue(torch.equal(state,torch.get_rng_state()))
            self.assertFalse(model.channel_context.last_debug['used_relative_norm'])
            stats = model.channel_context.last_debug['descriptor']
            s = x.std(-1,unbiased=False).flatten()
            d = (x[:,:,1:]-x[:,:,:-1]).abs().mean(-1).flatten()
            expected = torch.stack([s,torch.zeros_like(s),s,s,d,torch.zeros_like(d),d,d],1)
            self.assertTrue(torch.equal(stats,expected))
            model.queue_raw_context(x)
            with torch.no_grad():
                model.embedding(x+7)
            self.assertTrue(torch.equal(model.channel_context.last_debug['descriptor'],expected))
            self.assertEqual(model._queued_raw_context,[])

    def test_M_forward_no_relative_norm(self):
        with frozen_environment('M'):
            model = cac_variants.PatchEncoder(in_channels=3,use_revin=True).eval()
            self.assertEqual(sum(p.numel() for p in model.parameters()),896*3+375298)
            self.assertIsNone(model.dual_pooling_head)
            x = torch.randn(4,3,96)
            with torch.no_grad():
                self.assertEqual(tuple(model.embedding(x).shape),(4,64))
            self.assertFalse(model.channel_context.last_debug['used_relative_norm'])

    def test_frozen_jitter_formula_and_rng(self):
        with frozen_environment('U'):
            x = torch.randn(4,1,96)
            before = torch.get_rng_state().clone()
            actual = cross_branch_train.transferred_jitter(x)
            after = torch.get_rng_state().clone()
            torch.set_rng_state(before)
            mask = (torch.rand(4,1,1)<0.5).float()
            noise = 0.01*x.std(-1,keepdim=True,unbiased=False).clamp_min(1e-5)*torch.randn_like(x)
            expected = torch.nan_to_num(mask*(x+noise)+(1-mask)*x,nan=0.0,posinf=0.0,neginf=0.0)
            self.assertTrue(torch.equal(actual,expected))
            self.assertTrue(torch.equal(after,torch.get_rng_state()))

    def test_pooling_definition(self):
        from ha_paano._reference.model_FINAL_HA_PaAno import DualPoolingEmbeddingHead
        head = DualPoolingEmbeddingHead(embed_dim=64,init_alpha=0.05)
        x = torch.randn(4,64,96)
        h = head(x)
        alpha = torch.sigmoid(head.alpha_logit)
        avg, maximum = x.mean(-1), x.max(-1).values
        self.assertTrue(torch.equal(h,avg+alpha*(maximum-avg)))

    def test_data_loader_dry_run_and_overwrite_refusal(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'.release_audit') as temporary:
            base = Path(temporary)
            for track, channels in [('U',1),('M',3)]:
                folder = base/track
                folder.mkdir()
                path = folder/'001_TOY_id_1_Test_tr_128_1st_180.csv'
                data = np.random.randn(256,channels)
                labels = np.zeros(256); labels[180:196] = 1
                with path.open('w',newline='') as handle:
                    writer = csv.writer(handle)
                    writer.writerow([f'x{i}' for i in range(channels)]+['label'])
                    writer.writerows(np.column_stack([data,labels]))
                directory, rows = inspect_input(folder,track,'diagnostic')
                self.assertEqual(rows[0]['channels'],channels)
                train, train_labels, test, test_labels = load_and_split_data(path.as_posix())
                self.assertEqual(len(train),128)
                self.assertEqual(len(test),128)
                patches = preprocess_to_patches(np.array(train,dtype=np.float32),96,1)
                self.assertEqual(tuple(patches.shape),(33,channels,96))
                loaders = PatchCreator(96,1,2000).create_dataloaders(np.array(train),data,labels,batch_size=4)
                self.assertEqual(tuple(next(iter(loaders[0]))[0].shape),(4,channels,96))
                output = base/f'{track}_output'
                plan = plan_run(track,2000,folder,output,'diagnostic')
                self.assertFalse(output.exists())
                self.assertEqual(plan['task_count'],1)
                with self.assertRaises(ValueError):
                    plan_run(track,2000,folder,output,'formal')
                output.mkdir()
                with self.assertRaises(FileExistsError):
                    plan_run(track,2000,folder,output,'diagnostic')

    def test_score_alignment(self):
        patch_scores = np.array([0.1,0.3,0.5,0.7])
        actual = distribute_patch_scores_to_points(patch_scores,patch_size=3,num_points=6)
        expected = np.zeros(6); counts = np.zeros(6)
        for i,value in enumerate(patch_scores):
            expected[i:i+3] += value; counts[i:i+3] += 1
        self.assertTrue(np.allclose(actual,expected/counts))

    def test_metric_and_saver_cpu(self):
        labels = np.zeros(128); labels[40:52]=1; labels[90:98]=1
        score = np.linspace(0,0.3,128); score[40:52] += 0.6; score[90:98] += 0.3
        metrics = get_metrics(score,labels,slidingWindow=4,pred=None,version='opt',thre=250)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'.release_audit') as directory:
            path = save_one_file_result(directory,'toy.csv',np.zeros((128,1)),labels,score,metrics,0.1,{'test_only':True})
            with np.load(path,allow_pickle=False) as saved:
                self.assertTrue(np.array_equal(saved['scores'],score))
                self.assertTrue(np.array_equal(saved['labels'],labels))
                self.assertEqual(json.loads(str(saved['metrics'])),metrics)

    def test_unbound_pipeline_fails_closed(self):
        with self.assertRaises(RuntimeError):
            pipeline.PatchEncoder(in_channels=1)

    def test_public_execute_binding_without_training(self):
        from ha_paano.runner import execute_run
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            base=Path(temporary)
            data=base/'input.csv'; data.write_text('x,label\n0,0\n',encoding='utf-8')
            for track in ('U','M'):
                output=base/track
                plan={'track':track,'seed':2000,'scope':'diagnostic','config':read_config(track),
                      'data_dir':str(base),'output_dir':str(output),'series':[{'file':data.name}],
                      'task_count':1,'training_started':False}
                captured={}
                class DummyJob:
                    def __init__(self,**kwargs): captured.update(kwargs)
                    def run(self):
                        expected=cross_branch_model.PatchEncoder if track=='U' else cac_variants.PatchEncoder
                        if pipeline.PatchEncoder is not expected:
                            raise AssertionError('Wrong final model binding')
                old_binding=pipeline.PatchEncoder
                with patch.object(pipeline,'AnomalyDetection',DummyJob):
                    execute_run(plan,device='cpu')
                self.assertIs(pipeline.PatchEncoder,old_binding)
                self.assertEqual(captured['num_iters'],100)
                self.assertEqual(captured['batch_size'],512)
                self.assertEqual(captured['patch_size'],96)
                self.assertTrue(captured['use_revin'])
                self.assertEqual(json.loads((output/'run_receipt.json').read_text())['status'],'COMPLETE')
                self.assertFalse(torch.cuda.is_initialized())

    def test_readonly_result_inspector(self):
        from ha_paano._reference.utils.result_saver import append_summary_csv
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as temporary:
            root=Path(temporary)
            metrics={key:0.5 for key in ('VUS_PR','VUS_ROC','Point_F1','Range_F1','AUC_PR','AUC_ROC')}
            name='toy.csv'
            path=save_one_file_result(root/'results/npz',name,np.zeros((128,1)),np.zeros(128),np.ones(128),metrics,0.1)
            append_summary_csv(root/'results/summary.csv',{'file':name,**metrics})
            (root/'run_receipt.json').write_text(json.dumps({'series':[{'file':name}],'status':'COMPLETE'}))
            before=Path(path).read_bytes()
            result=subprocess.run([sys.executable,'-B','scripts/inspect_results.py','--run-dir',str(root)],
                                  cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('"evaluation_recomputed": false',result.stdout)
            self.assertEqual(before,Path(path).read_bytes())

if __name__ == '__main__':
    unittest.main(verbosity=2)
