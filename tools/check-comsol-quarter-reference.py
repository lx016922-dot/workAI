"""Audit reflected COMSOL eigenreferences and their actual spatial refinement."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from graphwave.modal_calibration import digest,match,read_modes,save
from spectral3d.mesh import boundary_data,make_mesh


def check_mesh(reference):
    raw=json.loads((Path(reference['folder'])/'scene.json').read_text())
    scene,cfg,receipt=raw['scene'],raw['execution'],reference['receipt']
    if cfg.get('domain_symmetry')!='quarter_yz' or receipt.get('domain_symmetry')!='quarter_yz':
        raise ValueError('An explicit quarter-domain reference is required')
    if int(receipt['physical_domain_multiplicity'])!=4 or receipt['mesh_type']!='hex':
        raise ValueError('Quarter-domain receipt mismatch')
    lengths=np.array([scene['geometry'][k] for k in ['length_mm','width_mm','thickness_mm']])*1e-3
    np.testing.assert_allclose([float(receipt['physical_width_m']),float(receipt['physical_thickness_m'])],lengths[1:],rtol=0,atol=1e-14)
    lengths[1:]*=.5;axes=[]
    for axis,length in enumerate(lengths):
        positions=np.unique(np.round(np.array(json.loads(receipt[f'mesh_axis_{axis}_m'])),13))
        # Match the actual Java double arithmetic in the declared mesher.
        steps=np.diff(positions);expected=int(np.ceil(length/(cfg['coarse_h_mm']*1e-3)))
        if len(steps)!=expected or np.any(steps<=0) or not np.allclose(positions[[0,-1]],[0,length],rtol=0,atol=1e-12):
            raise ValueError('Actual mesh axes do not form the declared quarter tensor grid')
        ratio=cfg.get('grading_ratio',1.) if axis==0 else cfg.get('cross_section_grading_ratio',1.)
        if ratio==1:
            correct=np.allclose(steps,steps[0],rtol=1e-7,atol=1e-12)
        elif axis==0:
            reverse=scene['boundary']['fixed_face']=='xmax'
            correct=np.all(np.diff(steps)<0) if reverse else np.all(np.diff(steps)>0)
        else:correct=np.all(np.diff(steps)>0)
        if not correct:raise ValueError('Mesh grading has an unintended direction')
        if len(steps)>1 and not np.isclose(steps.max()/steps.min(),ratio,rtol=1e-6,atol=1e-7):
            raise ValueError('Mesh grading ratio differs from the explicit input')
        axes.append({'axis':'xyz'[axis],'elements':len(steps),'coordinates_m':positions.tolist(),
                     'actual_spacing_ratio':float(steps.max()/steps.min()),'direction_verified':True})
    if np.prod([row['elements'] for row in axes])!=int(receipt['elements']):
        raise ValueError('Mesh is not the declared Cartesian tensor grid')
    points=reference['coordinates'];fields=reference['fields'];errors=[]
    dimensions=[len(np.unique(points[:,axis])) for axis in range(3)]
    index=np.arange(len(points)).reshape(dimensions)
    for axis in [1,2]:
        mirrored=np.flip(index,axis=axis).ravel();sign=np.ones(3);sign[axis]=-1
        residual=fields-fields[:,mirrored]*sign
        errors.append(float(np.max(np.linalg.norm(residual,axis=(1,2))/np.linalg.norm(fields,axis=(1,2)))))
    if max(errors)>1e-8:raise ValueError('Exported transverse fields do not have the physical reflection parity')
    return {'identity':reference['identity'],'axes':axes,'elements':int(receipt['elements']),
            'physical_domain_multiplicity':4,'maximum_reflection_error':max(errors),
            'mass_and_force_integral_scope':'quarter domain; force/mass ratio gives the full physical displacement',
            'actual_mesh_axes_verified':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--check-reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();a,b=read_modes(args.reference),read_modes(args.check_reference)
    for key in ['geometry','material','boundary','constitutive']:
        if a['scene'][key]!=b['scene'][key]:raise ValueError('Different physical problems')
    np.testing.assert_allclose(a['coordinates'],b['coordinates'],rtol=0,atol=1e-14)
    meshes=[check_mesh(r) for r in [a,b]];grid=make_mesh(a['scene'],4,2.5)
    mapping,mac=match(a['fields'],b['fields'],grid['mass']);load=boundary_data(grid,a['scene'])[0]
    participation=abs(np.einsum('mnc,nc->m',a['fields'],load))
    participation/=np.sqrt(np.einsum('mnc,mnc,n->m',a['fields'],a['fields'],grid['mass']))
    relevant=(a['frequency_hz']<=900000)&(participation>=.01*participation.max())
    difference=b['frequency_hz'][mapping]-a['frequency_hz'];maximum=float(abs(difference[relevant]).max())
    report={'schema':'workai.comsol_quarter_reference_check.v1','source_sha256':digest(__file__),
        'meshes':meshes,'energy_frequency_identity_verified':True,'phase_reference_gate_hz':.5,
        'maximum_load_relevant_difference_hz':maximum,
        'load_relevant_phase_gate_passed':bool(maximum<=.5 and mac[relevant].min()>=.99),
        'native_nonlinear_transient':False,'scientific_accuracy_certified':False,
        'rows':[{'mode_index':i,'fine_mode_index':int(mapping[i]),'load_relevant':bool(relevant[i]),
                 'coarse_frequency_hz':float(a['frequency_hz'][i]),'fine_frequency_hz':float(b['frequency_hz'][mapping[i]]),
                 'difference_hz':float(difference[i]),'mac':float(mac[i])} for i in range(len(mapping))]}
    save(args.output,report)
    print(json.dumps({k:report[k] for k in ['maximum_load_relevant_difference_hz','load_relevant_phase_gate_passed']}))


if __name__=='__main__':main()
