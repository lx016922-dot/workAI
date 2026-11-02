"""Check actual graded COMSOL meshes and their independent modal convergence."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from graphwave.modal_calibration import digest,match,read_modes
from spectral3d.mesh import boundary_data,make_mesh


def mesh_check(reference):
    scene=json.loads((Path(reference['folder'])/'scene.json').read_text())
    cfg=scene['execution'];receipt=reference['receipt'];axes=[]
    lengths=[scene['scene']['geometry'][key]*1e-3 for key in ['length_mm','width_mm','thickness_mm']]
    if scene['scene']['boundary']['fixed_face']!='xmax' or receipt['mesh_type']!='hex':
        raise ValueError('This diagnostic requires the declared fixed-end hex plate')
    for axis,length in enumerate(lengths):
        # Mesh vertices may contain roundoff-level duplicate coordinates.
        raw=np.asarray(json.loads(receipt[f'mesh_axis_{axis}_m']))
        values=np.unique(np.round(raw,13));steps=np.diff(values)
        expected=int(np.ceil(length*1e3/cfg['coarse_h_mm']))
        if len(steps)!=expected or not np.all(steps>0) or not np.allclose(values[[0,-1]],[0,length],atol=1e-12,rtol=0):
            raise ValueError('Exported mesh axes differ from the declared domain or count')
        correct=bool(steps[-1]<steps[0]) if axis==0 else bool(np.allclose(steps,steps[::-1],rtol=1e-8,atol=1e-12) and steps[len(steps)//2]>steps[0])
        if not correct:raise ValueError('Mesh grading points in an unintended direction')
        axes.append({'axis':'xyz'[axis],'coordinates_m':values.tolist(),'elements':len(steps),
                     'minimum_spacing_m':float(steps.min()),'maximum_spacing_m':float(steps.max()),
                     'largest_to_smallest_spacing':float(steps.max()/steps.min()),'direction_verified':correct})
    if np.prod([row['elements'] for row in axes])!=int(receipt['elements']):
        raise ValueError('Actual element count differs from the tensor grid')
    return {'identity':reference['identity'],'nominal_uniform_spacing_mm':cfg['coarse_h_mm'],
            'requested_grading_ratio':cfg['grading_ratio'],'elements':int(receipt['elements']),'axes':axes}


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--check-reference',type=Path,required=True);p.add_argument('--uniform',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    coarse,fine,uniform=[read_modes(path) for path in [args.reference,args.check_reference,args.uniform]]
    for other in [coarse,fine]:
        for key in ['geometry','material','boundary','constitutive']:
            if other['scene'][key]!=uniform['scene'][key]:raise ValueError('Physical reference scenes differ')
        np.testing.assert_allclose(other['coordinates'],uniform['coordinates'],rtol=0,atol=1e-14)
    meshes=[];mesh_errors=[]
    for value in [coarse,fine]:
        try:meshes.append(mesh_check(value))
        except ValueError as error:
            mesh_errors.append(str(error));meshes.append({'identity':value['identity'],'mesh_check_failed':str(error)})
    grid=make_mesh(uniform['scene'],4,2.5);load,_,_=boundary_data(grid,uniform['scene'])
    a,ma=match(uniform['fields'],coarse['fields'],grid['mass'])
    b,mb=match(uniform['fields'],fine['fields'],grid['mass'])
    participation=abs(np.einsum('mnc,nc->m',uniform['fields'],load))
    participation/=np.sqrt(np.einsum('mnc,mnc,n->m',uniform['fields'],uniform['fields'],grid['mass']))
    relevant=(uniform['frequency_hz']<=900000)&(participation>=.01*participation.max())
    difference=fine['frequency_hz'][b]-coarse['frequency_hz'][a]
    rows=[{'mode_index':i,'load_relevant':bool(relevant[i]),'relative_load_participation':float(participation[i]/participation.max()),
           'uniform_frequency_hz':float(uniform['frequency_hz'][i]),'graded_coarse_frequency_hz':float(coarse['frequency_hz'][a[i]]),
           'graded_fine_frequency_hz':float(fine['frequency_hz'][b[i]]),'graded_refinement_difference_hz':float(difference[i]),
           'coarse_to_uniform_mac':float(ma[i]),'fine_to_uniform_mac':float(mb[i])} for i in range(len(difference))]
    primary=int(np.argmax(participation));maximum=float(abs(difference[relevant]).max())
    report={'schema':'workai.comsol_graded_reference_check.v1','source_sha256':digest(__file__),
            'graded_meshes':meshes,'uniform_identity':uniform['identity'],'primary_mode_index':primary,'primary_mode':rows[primary],
            'load_relevant_modes':np.flatnonzero(relevant).tolist(),'maximum_load_relevant_difference_hz':maximum,
            'phase_reference_gate_hz':.5,'load_relevant_phase_gate_passed':bool(not mesh_errors and maximum<=.5 and min(ma[relevant].min(),mb[relevant].min())>=.99),
            'actual_mesh_axes_verified':not mesh_errors,'mesh_errors':mesh_errors,
            'energy_frequency_identity_verified':True,'used_for_v8_training':False,
            'candidate_waveforms_used_for_selection':False,'native_nonlinear_transient':False,'scientific_accuracy_certified':False,'rows':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ['primary_mode','maximum_load_relevant_difference_hz','load_relevant_phase_gate_passed']}))


if __name__=='__main__':main()
