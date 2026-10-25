"""Compile a research modal basis from candidate weights alone, with provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from graphwave.graded_mesh import make_graded_mesh
from graphwave.modal_calibration import modes
from graphwave.modal_reduction import axial_reflection_subspace,attach_basis

SOURCES=['graphwave/graded_mesh.py','graphwave/modal_calibration.py','graphwave/modal_reduction.py',
         'graphwave/modal_sensitivity.py','graphwave/buffered_inference.py','graphwave/inference.py',
         'graphwave/signals.py','graphwave/modal_sparse.py','graphwave/quarter_symmetry.py',
         'spectral3d/mesh.py','tools/compile-modal-candidate.py']

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--degree',type=int,choices=[4,5,6],default=6);p.add_argument('--h-mm',type=float,required=True)
    p.add_argument('--axis-elements',type=int,nargs=3,required=True);p.add_argument('--grading-ratios',type=float,nargs=3,required=True)
    p.add_argument('--quarter-domain',action='store_true')
    p.add_argument('--modes',type=int,default=200);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();repo=Path(__file__).resolve().parents[1]
    sources={name:digest(repo/name) for name in SOURCES}
    raw=json.loads(args.input.read_text());scene=raw.get('scene',raw)
    weights_hash=digest(args.weights)
    with np.load(args.weights,allow_pickle=False) as saved:weights=dict(saved)
    if args.quarter_domain:
        from graphwave.quarter_symmetry import make_quarter_mesh
        mesh=make_quarter_mesh(scene,args.degree,args.h_mm,args.grading_ratios,args.axis_elements)
    else:mesh=make_graded_mesh(scene,args.degree,args.h_mm,args.grading_ratios,args.axis_elements)
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    for name in SOURCES:
        destination=args.output/'source_snapshot'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(repo/name,destination)
    frequency,fields=modes(mesh,scene,weights['derivative_'+str(args.degree)],weights['quadratic_tensor'],args.modes)
    if args.quarter_domain:
        selected=np.arange(len(frequency));symmetry={'method':'quarter-domain y/z reflection with component-wise essential constraints',
            'original_modes':len(frequency),'retained_modes':len(frequency),'physical_domain_multiplicity':4,
            'transverse_displacements_preserved_with_reflection_signs':True,'requires_full_basis_nonlinear_check':True}
    else:selected,symmetry=axial_reflection_subspace(mesh,scene,weights,fields)
    model=attach_basis(mesh,scene,weights,frequency[selected],fields[selected])
    if sources!={name:digest(repo/name) for name in SOURCES} or weights_hash!=digest(args.weights):
        raise ValueError('Source or weights changed during compilation; basis is not exportable')
    np.savez_compressed(args.output/'basis.npz',frequency_hz=frequency,fields=fields)
    shutil.copyfile(args.weights,args.output/'weights.npz')
    result={'schema':'workai.graph_modal_candidate.v1','scene':scene,'degree':args.degree,'h_mm':args.h_mm,
            'axis_elements':args.axis_elements,'grading_ratios':args.grading_ratios,'exported_modes':args.modes,
            'quarter_domain':args.quarter_domain,
            'nodes':len(mesh['points']),'elements':len(mesh['elements']),
            'weights_sha256':weights_hash,'basis_sha256':digest(args.output/'basis.npz'),
            'compilation_source_sha256':sources,'symmetry':symmetry,
            'mass_orthonormality_error':model['mass_orthonormality_error'],
            'compilation_seconds':time.monotonic()-started,'comsol_arrays_used':False,
            'numerical_prediction_corrector':False,'material_tensor_entries_dropped':False,
            'modal_truncation_convergence_verified':False,'scientific_accuracy_certified':False}
    (args.output/'model.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ['exported_modes','nodes','elements','compilation_seconds','weights_sha256','basis_sha256']}),flush=True)

if __name__=='__main__':main()
