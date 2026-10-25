import copy,hashlib,importlib,io,json,sys,tempfile,unittest,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.reproduce_tables import build
from scripts.audit_publication import inspect
from scripts.fetch_artifacts import extract
from src.deployment_probe_v1 import axis_features,required_views
from src.deployment_guard_v1 import guard_proposal
from src.mechanism_v3 import conditional_messages_v3

class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables=build();cls.policy=json.loads((ROOT/'configs/frozen_policy.json').read_text())
        cls.cases=[json.loads(l) for l in (ROOT/'data/conditional/cases.jsonl').read_text(encoding='utf-8').splitlines()]
    def test_source_bytes_match_formal_run(self):
        manifest=json.loads((ROOT/'resources/source_manifest.json').read_text())
        for r in manifest['files']:self.assertEqual(hashlib.sha256((ROOT/r['public_path']).read_bytes()).hexdigest(),r['sha256'],r['public_path'])
    def test_every_module_imports_without_loading_models(self):
        for p in (ROOT/'src').glob('*.py'):importlib.import_module('src.'+p.stem)
    def test_policy_retains_full_precision(self):
        self.assertEqual(self.policy['alpha_arg'],2);self.assertEqual(self.policy['theta_name'],0.02706692410243683)
        manifest=json.loads((ROOT/'resources/source_manifest.json').read_text())
        self.assertEqual(hashlib.sha256((ROOT/'configs/frozen_policy.json').read_bytes()).hexdigest(),manifest['frozen_policy_sha256'])
    def test_real_controlled_data_are_complete(self):
        self.assertEqual(len(self.cases),600);self.assertEqual(len({c['case_id'] for c in self.cases}),600)
        self.assertEqual(len({c['variant'] for c in self.cases}),6)
    def test_axes_do_not_mix_the_two_alphas(self):
        s={v:{'name':{'length_normalized_loglik':n},'full':{'length_normalized_loglik':f}} for v,n,f in [('original',.2,.8),('user_masked',.7,1.7),('observation_masked',.3,.6)]}
        f=axis_features(s,.5,2.);self.assertAlmostEqual(f['Risk_name'],.15);self.assertAlmostEqual(f['DeltaRisk_arg'],1.1)
        self.assertEqual(len(required_views(1.,2.)),3)
    def test_scope_and_discovery_do_not_score(self):
        p={'kind':'tool_call','tool':'get_channels','arguments':{}}
        def forbidden(*args):raise AssertionError('Unexpected forward')
        self.assertEqual(guard_proposal(p,[],{'get_channels':{'parameters':[]}},scope_allowed=True,probe=forbidden)['extra_forward_calls'],0)
        self.assertEqual(guard_proposal(p,[],{'get_channels':{'parameters':[]}},scope_allowed=False,probe=forbidden)['decision'],'block_scope')
    def test_missing_evidence_stops_before_inference(self):
        from examples.quick_start import main
        self.assertEqual(main()['reason'],'grounding')
    def test_table_reconstruction_matches_accepted_counts(self):
        r=self.tables['table_09_standard'];self.assertEqual([x['confirmed'] for x in r],[83,83,0]);self.assertEqual([x['utility'] for x in r],[98,126,109]);self.assertEqual([x['legal_retained'] for x in r],[47,47,30])
        self.assertEqual([x['injecagent_count'] for x in self.tables['table_04_v1']],[107,119,107,107,107,31])
        self.assertEqual([x['rejected'] for x in self.tables['calibration_events']],[2,104,21])
    def test_prefix_auc_uses_three_equal_weight_groups(self):
        r=self.tables['table_05_prefix'];self.assertAlmostEqual(r[0]['macro_auc'],.7163831716261384)
        self.assertEqual([x['macro_auc'] for x in r if x['prefix']!='with_actual_thought'],[1.,1.,1.,1.])
    def test_sign_tests_use_domains(self):
        for r in self.tables['table_06_reversal_tests']:self.assertEqual((r['families'],r['effective_domains'],r['one_sided_p']),(10,5,.03125))
    def test_publication_guard_blocks_renamed_manuscript_and_tokens(self):
        p=json.loads((ROOT/'resources/publication_policy.json').read_text())
        self.assertTrue(inspect('innocent.txt',b'\\document'+b'class{article}',p))
        self.assertTrue(inspect('paper/draft.md',b'draft',p))
        self.assertTrue(inspect('config.txt',('ghp_'+'A'*36).encode(),p))
        self.assertFalse(inspect('licenses/third-party.txt',b'MIT License',p))
    def test_archive_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'bad.zip';b=b'bad'
            with zipfile.ZipFile(path,'w') as z:z.writestr('../escape',b)
            manifest={'files':{'../escape':{'sha256':hashlib.sha256(b).hexdigest(),'bytes':3}}}
            with self.assertRaises(ValueError):extract(path,manifest,Path(temp)/'output')

if __name__=='__main__':unittest.main()
