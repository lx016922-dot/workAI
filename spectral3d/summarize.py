"""Bind completed reference checks to a read-only progress and label manifest."""
import argparse
import json
from pathlib import Path
from .campaign import atomic,now
from .compare import digest,spectral


def main(args):
    evidence=args.evidence
    paths={'h':evidence/'p4-h16-h10.json','p':evidence/'p4-p5-h10.json','time':evidence/'p4-h10-time.json','field':evidence/'p4-h16-h10-fields.json'}
    reports={k:json.loads(p.read_text()) for k,p in paths.items()}
    campaigns={name:json.loads((args.root/'spectral3d/campaigns'/name/'manifest.json').read_text()) for name in args.campaigns}
    assert all(c['phase']=='completed' for c in campaigns.values())
    hashes={digest(args.root/'spectral3d/campaigns'/name/'manifest.json') for name in campaigns}
    for report in reports.values():assert report['left_manifest_sha256'] in hashes and report['right_manifest_sha256'] in hashes
    fine_path=args.root/'spectral3d/campaigns'/args.fine/'manifest.json';fine=json.loads(fine_path.read_text());validated=spectral(fine_path);rows=[];labels=[]
    fine_sha=digest(fine_path)
    assert reports['h']['right_manifest_sha256']==reports['field']['right_manifest_sha256']==reports['p']['left_manifest_sha256']==reports['time']['left_manifest_sha256']==fine_sha
    assert reports['h']['left_manifest_sha256']==reports['field']['left_manifest_sha256']
    for case in fine['cases']:
        checks={key:next(r for r in report['cases'] if r['case_id']==case['case_id']) for key,report in reports.items()}
        maximum=lambda kind,key:max(p['checks'][key]['relative_l2'] for p in checks[kind]['probes'])
        energy=all(d['energy_balance_within_1_percent'] and d['max_strain']<=.001 and d['min_J']>0 for d in validated[case['case_id']]['diagnostics'])
        row={'case_id':case['case_id'],'h_passed':checks['h']['all_probe_checks_pass'],'p_passed':checks['p']['all_probe_checks_pass'],'time_passed':checks['time']['all_probe_checks_pass'],
             'field_passed':checks['field']['all_checks_pass'],'energy_and_strain_passed':energy,
             'max_h_full_vector':maximum('h','full_vector'),'max_h_material_vector':maximum('h','material_vector'),
             'max_p_material_vector':maximum('p','material_vector'),'max_time_material_vector':maximum('time','material_vector'),
             'field_h_full_vector':checks['field']['checks']['full_vector']['relative_l2'],'field_h_material_vector':checks['field']['checks']['material_vector']['relative_l2']}
        row['all_component_checks_passed']=all(row[k] for k in ['h_passed','p_passed','time_passed','field_passed','energy_and_strain_passed']);rows.append(row)
        labels.append({'case_id':case['case_id'],'split':'development_consumed','scene':case['scene'],'mesh_file':case['mesh_file'],'mesh_sha256':case['mesh_sha256'],
                       'controls':case['controls'],'reference_checks_passed':row['all_component_checks_passed'],'routing_accepted':False})
    summary={'schema':'workai.spectral_reference_summary.v1','completed_at_utc':now(),'completed_solves':sum(len(row['controls']) for c in campaigns.values() for row in c['cases']),
        'threshold':.05,'reporting_floor_nm':.0001,'cases':rows,'all_checks_passed':all(r['all_component_checks_passed'] for r in rows),
        'training_used_these_fields':False,'accepted_for_solver_routing':False,'new_training_steps':0,
        'evidence_sha256':{key:digest(path) for key,path in paths.items()},'campaign_sha256':{name:digest(args.root/'spectral3d/campaigns'/name/'manifest.json') for name in campaigns},
        'scope':'Plate and through-width notch only. Three consumed geometries. h checks include discrete whole-field RMS; p and time checks cover retained receivers. No arbitrary-point error bound, independent blind AI evaluation, pipe/hole convergence or COMSOL comparison.'}
    label_manifest={'schema':'workai.spectral_development_labels.v1','created_at_utc':now(),'source_campaign_sha256':digest(fine_path),'case_count':len(labels),
        'training_used_these_fields':False,'accepted_for_solver_routing':False,'cases':labels,'summary_sha256':None,
        'admission_rule':'All retained-component h/p/time checks, full-field discrete h check and energy/strain guards must pass; these are development labels, never fresh test data.'}
    atomic(evidence/'summary.json',summary);label_manifest['summary_sha256']=digest(evidence/'summary.json');atomic(evidence/'development-labels.json',label_manifest)
    atomic(args.root/'spectral3d/latest.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/workspace/workai-remote'));p.add_argument('--evidence',type=Path,required=True);p.add_argument('--campaigns',nargs='+',required=True);p.add_argument('--fine',required=True);main(p.parse_args())
