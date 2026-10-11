"""CPU tests of real tiny Qwen/PEFT, without downloading weights or datasets."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import torch
from protocol import assert_separate, coverage_curve, epoch_batches, learning_rate, length_weights, target_probability, training_weights, vote_probability
from train import adapter_state, decision_scores, evaluate, load_model, make_optimizer, restore_checkpoint, save_checkpoint, train_update


def row(index, target='A'):
    return {'id':str(index),'group_id':str(index),'target':target,'strength':2,'shorter_won':index%2==0,
            'multi_turn':False,'vote_p_a':2/3,'tokens':{'json':[[1,2,3,4],[1,2,4,3]]}}


class ProtocolTests(unittest.TestCase):
    def test_epoch_order_keeps_pairs_and_is_restartable(self):
        import random
        rows = [row(i) for i in range(11)]
        state = random.getstate()
        batches = epoch_batches(rows, 2, 123)
        self.assertEqual(state, random.getstate())
        first = [r['id'] for b in batches[:3] for r in b]
        second = [r['id'] for b in batches[3:] for r in b]
        self.assertEqual(first, [r['id'] for r in rows])
        self.assertCountEqual(first, second)
        self.assertNotEqual(first, second)
        self.assertEqual(batches[4:], epoch_batches(rows, 2, 123)[4:])
        self.assertEqual(batches[:3], epoch_batches(rows, 2, 123, False)[3:])

    def test_strength_weights_preserve_ratios_and_mean_scale(self):
        rows = [row(i) | {'strength': i + 1} for i in range(3)]
        weights = training_weights(rows, strength=True)
        self.assertEqual(weights, {'0': .5, '1': 1., '2': 1.5})
        self.assertEqual(training_weights(rows), length_weights(rows, False))
        with self.assertRaises(ValueError):
            training_weights([row(0, 'tie_preference')], strength=True)

    def test_larger_monitor_preserves_groups_and_existing_pool(self):
        from prepare_curve import enlarge_monitor
        training = [row(i) for i in range(60)]
        training.append(row(99, 'tie_preference') | {'group_id': '20'})
        old = [row(70), row(71)]
        pool = [str(i) for i in range(10)]
        remaining, monitor = enlarge_monitor(training, old, pool, 12, 123)
        self.assertEqual(len(monitor), 12)
        self.assertEqual(monitor[:2], old)
        self.assertTrue(set(pool) <= {r['id'] for r in remaining})
        assert_separate(remaining, monitor, [])
        self.assertEqual((remaining, monitor), enlarge_monitor(training, old, pool, 12, 123))
        with self.assertRaises(ValueError): enlarge_monitor(training, old, pool, 200, 123)

    def test_votes_and_swaps(self):
        self.assertEqual(vote_probability([{'strength':-1},{'strength':2},{'strength':-3}]),2/3)
        for method in ('hard','smooth','votes'):
            self.assertAlmostEqual(target_probability(row(0),method)+target_probability(row(0),method,True),1)
        self.assertAlmostEqual(target_probability(row(0),'smooth'),.95)
        with self.assertRaises(ValueError): target_probability(row(0,'tie_preference'),'hard')
        with self.assertRaises(ValueError): vote_probability([{'strength':0},{'strength':1}])

    def test_groups_and_weights(self):
        with self.assertRaises(ValueError): assert_separate([row(0)],[row(0)],[row(2)])
        rows=[row(0),row(1),row(3)]
        weights=length_weights(rows,True)
        self.assertAlmostEqual(weights['0'],weights['1']+weights['3'])
        assert_separate([row(0)],[row(1)],[row(2)])

    def test_schedule_and_coverage(self):
        self.assertAlmostEqual(learning_rate(0,2e-5,50,150),4e-7)
        self.assertAlmostEqual(learning_rate(49,2e-5,50,150),2e-5)
        self.assertAlmostEqual(learning_rate(149,2e-5,50,150),2e-6)
        pairs=[row(i)|{'p_a':.8} for i in range(10)]
        self.assertTrue(all(p['actual_coverage']==1 for p in coverage_curve(pairs)))
        with self.assertRaises(ValueError): coverage_curve([])

    def test_adam_and_clipping(self):
        model=torch.nn.Linear(1,1,bias=False)
        with torch.no_grad(): model.weight.fill_(2)
        optimizer=make_optimizer(model)
        optimizer.param_groups[0]['lr']=.01
        model.weight.grad=torch.tensor([[5000.]])
        raw=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        self.assertEqual(float(raw),5000)
        self.assertLessEqual(float(model.weight.grad.norm()),1.000001)
        optimizer.step()
        self.assertAlmostEqual(float(model.weight.detach()),1.99,places=6)

    def test_fp32_head(self):
        class Body(torch.nn.Module):
            def forward(self, **kwargs):
                from types import SimpleNamespace
                return SimpleNamespace(last_hidden_state=torch.ones(1,1,2,dtype=torch.bfloat16))
        class Fake(torch.nn.Module):
            def __init__(self):
                super().__init__();self.model=Body();self.embedding=torch.nn.Embedding(64,2,dtype=torch.bfloat16)
                with torch.no_grad():
                    self.embedding.weight[32]=torch.tensor([21.,.03125])
                    self.embedding.weight[33]=torch.tensor([21.,.0625])
            def get_output_embeddings(self):return self.embedding
        model=Fake()
        result=decision_scores(model,torch.tensor([[1]]))
        self.assertEqual(result.dtype,torch.float32)
        self.assertEqual(result.tolist(),[[21.03125,21.0625]])
        rounded=torch.ones(1,2,dtype=torch.bfloat16) @ model.embedding.weight[[32,33]].T
        self.assertEqual(float(rounded[0,0].detach()),float(rounded[0,1].detach()))

    def test_tiny_qwen_resume_with_dropout_and_evaluation(self):
        recipe={'seed':123,'layers':36,'learning_rate':2e-5,'warmup':2,'updates':5,'format':'json','labels':'votes'}
        rows=[row(i) for i in range(4)];weights=length_weights(rows,False)
        model=load_model(recipe,tiny=True);optimizer=make_optimizer(model)
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            train_update(model,optimizer,rows,weights,recipe,0)
            save_checkpoint(folder/'checkpoint',model,optimizer,1,recipe)
            evaluate(model,rows,'json',folder/'evaluation')
            next_entry=train_update(model,optimizer,rows,weights,recipe,1)
            expected=adapter_state(model)
            expected_opt=copy.deepcopy(optimizer.state_dict())
            restarted=load_model(recipe,tiny=True);restarted_opt=make_optimizer(restarted)
            self.assertEqual(restore_checkpoint(folder/'checkpoint',restarted,restarted_opt,recipe),1)
            actual_entry=train_update(restarted,restarted_opt,rows,weights,recipe,1)
            self.assertEqual(next_entry['mean_unweighted_loss'],actual_entry['mean_unweighted_loss'])
            self.assertEqual(next_entry['raw_gradient_norm'],actual_entry['raw_gradient_norm'])
            self.assertTrue(all(torch.equal(p,adapter_state(restarted)[name]) for name,p in expected.items()))
            for i,state in expected_opt['state'].items():
                self.assertTrue(all(torch.equal(value,restarted_opt.state_dict()['state'][i][key]) for key,value in state.items()))
            with self.assertRaises(ValueError): restore_checkpoint(folder/'checkpoint',restarted,restarted_opt,recipe|{'seed':456})
            with self.assertRaises(ValueError): save_checkpoint(folder/'checkpoint',model,optimizer,1,recipe)


    def test_six_votes_are_preserved(self):
        self.assertEqual(vote_probability([{'strength':-1}]*4+[{'strength':1}]*2),2/3)

    def test_metrics_known_ranking_and_paired_changes(self):
        import numpy as np
        from metrics import weighted_metrics, mcnemar
        measured=weighted_metrics(np.array([0.,1.]),np.array([.1,.9]),np.ones((1,2)))[0]
        self.assertEqual(measured[0],1)
        self.assertAlmostEqual(measured[1],-__import__('math').log(.9))
        self.assertEqual(measured[2],1)
        self.assertEqual(mcnemar([False]*6,[True]*6)['exact_two_sided_p'],.03125)

    def test_worker_resume_keeps_interrupted_evidence(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from protocol import save_json, sha256
        from train import execute
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle=root/'bundle';bundle.mkdir()
            for name,rows in [('training',[row(i) for i in range(20)]),('monitor',[row(30),row(31)])]:
                (bundle/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            save_json(bundle/'manifest.json',{'files_sha256':{p.name:sha256(p) for p in bundle.iterdir()},
                'validation_groups':['validation-only'],'pool2000_ids':[str(i) for i in range(20)],'model':'tiny','revision':'test'})
            args=SimpleNamespace(bundle=str(bundle),output=str(root/'interrupted'),learning_rate=2e-5,updates=10,
                warmup=2,epochs=2,labels='hard',style='json',pool='pool2000',layers=36,balance_length=False,
                seed=20261007,shuffle_epochs=True,max_pairs=None,weight_strength=False,minimum_epochs=0,
                resume=None,monitor_every=1,checkpoint_every=25,wall_minutes=10,patience=0,minimum_improvement=.002)
            def fail_second_epoch(*values):
                if values[-1]==6:raise RuntimeError('Test interruption during the shuffled second epoch.')
                return train_update(*values)
            with patch('train.load_model',side_effect=lambda recipe:load_model(recipe,tiny=True)):
                with patch('train.train_update',side_effect=fail_second_epoch):
                    with self.assertRaisesRegex(RuntimeError,'Test interruption'):execute(args)
                args.resume=str(root/'interrupted/checkpoint_0006')
                execute(args)
                uninterrupted = copy.copy(args)
                uninterrupted.output = str(root/'continuous'); uninterrupted.resume = None
                execute(uninterrupted)
                minimum_pass = copy.copy(uninterrupted)
                minimum_pass.output = str(root/'minimum_pass')
                minimum_pass.patience = 1; minimum_pass.minimum_improvement = 100
                minimum_pass.minimum_epochs = 1
                execute(minimum_pass)
                self.assertEqual(json.loads((root/'minimum_pass/result.json').read_text())['completed_updates'], 5)
            resumed = torch.load(root/'interrupted/checkpoint_0010/resume.pt', weights_only=True)
            continuous = torch.load(root/'continuous/checkpoint_0010/resume.pt', weights_only=True)
            self.assertTrue(all(torch.equal(value, continuous['adapter'][name]) for name, value in resumed['adapter'].items()))
            for i, state in resumed['optimizer']['state'].items():
                self.assertTrue(all(torch.equal(value, continuous['optimizer']['state'][i][key]) for key, value in state.items()))
            self.assertTrue(torch.equal(resumed['rng']['cpu'], continuous['rng']['cpu']))
            result=json.loads((root/'interrupted/result.json').read_text())
            self.assertEqual(result['status'],'completed')
            self.assertEqual([r['step'] for r in result['monitors']],list(range(1,11)))
            self.assertTrue(list((root/'interrupted').glob('interrupted_*/updates.jsonl')))
            self.assertFalse(result['validation_opened'])


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
