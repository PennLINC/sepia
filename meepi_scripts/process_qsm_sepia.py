"""Prepare ME-EPI QSM inputs and submit SEPIA MATLAB array jobs.

The expected input layout is one 4D BOLD file per echo and part, for example:

    sub-01_echo-1_part-mag_bold.nii.gz
    sub-01_echo-1_part-phase_bold.nii.gz

For each run, this script:

1. Collects echo-wise magnitude and phase BOLD files.
2. Unwraps all echo-wise phase files with one ``wk-unwrap-phase`` call.
3. Splits the 4D echo-wise files by time point and writes volume-wise
   multi-echo magnitude/phase files for SEPIA.
4. Writes one configured MATLAB script per volume and submits them as a
   SLURM array job, capped to 10 concurrent MATLAB tasks by default.
5. Submits a dependent job that concatenates volume-wise SEPIA chi maps
   back into one 4D image per run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from pprint import pformat

import nibabel as nb
import numpy as np
from bids.layout import BIDSLayout, Query
from scipy.io import loadmat, savemat

CFG = {
    'bids_dir': '/cbica/projects/pafin/dset',
    'code_dir': '/cbica/projects/pafin/projects/qsm-validation/code',
    'work_dir': '/cbica/comp_space/pafin/qsm-validation',
    'derivatives': {
        'meepi': '/cbica/projects/pafin/projects/qsm-validation/derivatives/meepi',
    }
}
CODE_DIR = CFG['code_dir']


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _drop_empty_entities(entities):
    return {key: val for key, val in entities.items() if val not in (None, '', Query.NONE)}


def _sort_by_echo(files):
    return sorted(files, key=lambda f: int(f.get_entities().get('echo', 0)))


def collect_run_data(layout: BIDSLayout, bids_filters: dict) -> dict:
    """Collect one ME-EPI run for SEPIA QSM estimation."""
    bold_base_query = {
        'datatype': 'func',
        'echo': Query.ANY,
        'space': Query.NONE,
        'desc': Query.NONE,
        'suffix': 'bold',
        'extension': ['.nii', '.nii.gz'],
    }
    mag_query = _drop_empty_entities({**bids_filters, **bold_base_query, 'part': 'mag'})
    phase_query = _drop_empty_entities({**bids_filters, **bold_base_query, 'part': 'phase'})

    mag_files = _sort_by_echo(layout.get(**mag_query))
    phase_files = _sort_by_echo(layout.get(**phase_query))
    if not mag_files:
        raise ValueError(f'No magnitude BOLD files found with query {mag_query}')
    if not phase_files:
        raise ValueError(f'No phase BOLD files found with query {phase_query}')

    mag_echoes = [int(f.get_entities()['echo']) for f in mag_files]
    phase_echoes = [int(f.get_entities()['echo']) for f in phase_files]
    if mag_echoes != phase_echoes:
        raise ValueError(f'Magnitude echoes {mag_echoes} do not match phase echoes {phase_echoes}')

    run_data = {
        'bold_mag': [f.path for f in mag_files],
        'bold_phase': [f.path for f in phase_files],
    }
    print(pformat(run_data), flush=True)
    return run_data


def get_echo_times(layout: BIDSLayout, echo_files: list[str], header_struct: dict) -> np.ndarray:
    """Get echo times from BIDS metadata, falling back to the SEPIA header."""
    echo_times = []
    for echo_file in echo_files:
        metadata = layout.get_metadata(echo_file)
        echo_time = metadata.get('EchoTime')
        if echo_time is None:
            echo_times = []
            break
        echo_times.append(float(echo_time))

    if echo_times:
        return np.asarray(echo_times, dtype=float).reshape(1, -1)

    if 'TE' in header_struct and header_struct['TE'].size >= len(echo_files):
        return np.asarray(header_struct['TE'], dtype=float).reshape(1, -1)[:, : len(echo_files)]

    raise ValueError('Could not determine echo times from BIDS metadata or sepia_header.mat')


def _sort_wk_outputs(files: list[str]) -> list[str]:
    """Sort wk-unwrap-phase outputs by echo label when possible."""
    def sort_key(path):
        name = os.path.basename(path)
        match = re.search(r'echo-?(\d+)', name)
        if match:
            return (int(match.group(1)), name)
        return (10**6, name)

    return sorted(files, key=sort_key)


def unwrap_phase_data(
    run_data: dict,
    temp_dir: str,
    echo_times: np.ndarray,
    overwrite: bool,
    n_cpus,
    extra_args: str,
) -> list[str]:
    """Unwrap all echo-wise phase files with wk-unwrap-phase."""
    unwrap_dir = os.path.join(temp_dir, 'wk_unwrap')
    os.makedirs(unwrap_dir, exist_ok=True)
    out_prefix = os.path.join(unwrap_dir, 'desc-wkunwrap')

    existing_outputs = _sort_wk_outputs(
        [
            os.path.join(unwrap_dir, name)
            for name in os.listdir(unwrap_dir)
            if name.startswith('desc-wkunwrap')
            and name.endswith(('.nii', '.nii.gz'))
            and 'mask' not in name.lower()
        ]
    )
    if len(existing_outputs) == len(run_data['bold_phase']) and not overwrite:
        return existing_outputs

    for name in os.listdir(unwrap_dir):
        if name.startswith('desc-wkunwrap'):
            os.remove(os.path.join(unwrap_dir, name))

    tes_ms = [f'{te * 1000:g}' for te in np.ravel(echo_times)]
    cmd = [
        'wk-unwrap-phase',
        '--magnitude',
        *run_data['bold_mag'],
        '--phase',
        *run_data['bold_phase'],
        '--TEs',
        *tes_ms,
        '--out-prefix',
        out_prefix,
    ]
    if n_cpus is not None:
        cmd.extend(['--n-cpus', str(n_cpus)])
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    print(f'Running phase unwrapping: {shlex.join(cmd)}', flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f'wk-unwrap-phase failed with code {result.returncode}\n'
            f'Command: {shlex.join(cmd)}\n'
            f'stdout:\n{result.stdout}\n'
            f'stderr:\n{result.stderr}'
        )

    unwrapped_phase_files = _sort_wk_outputs(
        [
            os.path.join(unwrap_dir, name)
            for name in os.listdir(unwrap_dir)
            if name.startswith('desc-wkunwrap')
            and name.endswith(('.nii', '.nii.gz'))
            and 'mask' not in name.lower()
        ]
    )
    if len(unwrapped_phase_files) != len(run_data['bold_phase']):
        raise FileNotFoundError(
            'wk-unwrap-phase did not produce one unwrapped phase NIfTI per echo. '
            f'Expected {len(run_data["bold_phase"])}, found {len(unwrapped_phase_files)}: '
            f'{unwrapped_phase_files}'
        )
    return unwrapped_phase_files


def _load_volume(img: nb.spatialimages.SpatialImage, volume_idx: int) -> np.ndarray:
    if img.ndim == 3:
        if volume_idx != 0:
            raise IndexError('Cannot load volume > 0 from a 3D image')
        return np.asanyarray(img.dataobj)
    return np.asanyarray(img.dataobj[..., volume_idx])


def write_volume_inputs(
    run_data: dict,
    unwrapped_phase_files: list[str],
    temp_dir: str,
    header_struct: dict,
    echo_times: np.ndarray,
) -> list[dict]:
    """Write volume-wise multi-echo magnitude/phase/header inputs for SEPIA."""
    input_dir = os.path.join(temp_dir, 'sepia_inputs')
    os.makedirs(input_dir, exist_ok=True)

    mag_imgs = [nb.load(f) for f in run_data['bold_mag']]
    phase_imgs = [nb.load(f) for f in unwrapped_phase_files]
    shapes = [img.shape for img in [*mag_imgs, *phase_imgs]]
    if len(set(shapes)) != 1:
        raise ValueError(f'All magnitude and phase files must have the same shape. Got {shapes}')

    n_volumes = shapes[0][3] if len(shapes[0]) == 4 else 1
    volume_inputs = []
    for volume_idx in range(n_volumes):
        mag_data = np.stack([_load_volume(img, volume_idx) for img in mag_imgs], axis=3)
        phase_data = np.stack([_load_volume(img, volume_idx) for img in phase_imgs], axis=3)

        volume_label = f'vol-{volume_idx + 1:04d}'
        mag_file = os.path.join(input_dir, f'{volume_label}_part-mag_bold.nii.gz')
        phase_file = os.path.join(input_dir, f'{volume_label}_part-phase_desc-wkunwrap_bold.nii.gz')
        header_file = os.path.join(input_dir, f'{volume_label}_sepia_header.mat')

        ref_img = mag_imgs[0]
        nb.Nifti1Image(mag_data, ref_img.affine, ref_img.header).to_filename(mag_file)
        nb.Nifti1Image(phase_data, ref_img.affine, ref_img.header).to_filename(phase_file)

        volume_header = dict(header_struct)
        volume_header['TE'] = echo_times
        if echo_times.shape[1] > 1:
            volume_header['delta_TE'] = np.asarray([[echo_times[0, 1] - echo_times[0, 0]]])
        savemat(header_file, volume_header)

        volume_inputs.append(
            {
                'volume_idx': volume_idx + 1,
                'mag_file': mag_file,
                'phase_file': phase_file,
                'header_file': header_file,
            }
        )

    return volume_inputs


def _matlab_literal_path(path: str) -> str:
    return "'" + path.replace("'", "''") + "'"


def write_matlab_scripts(
    layout: BIDSLayout,
    run_data: dict,
    volume_inputs: list[dict],
    out_dir: str,
    temp_dir: str,
) -> tuple[list[str], dict]:
    """Write one configured MATLAB script per volume."""
    sepia_script = os.path.join(CODE_DIR, 'processing', 'process_qsm_sepia.m')
    if not os.path.isfile(sepia_script):
        sepia_script = os.path.join(Path(__file__).resolve().parent, 'process_qsm_sepia.m')
    with open(sepia_script) as fobj:
        base_sepia_script = fobj.read()

    script_paths = []
    chimap_files = []
    name_source = run_data['bold_mag'][0]
    base_name = os.path.basename(name_source)
    final_4d_chimap_file = os.path.join(out_dir, base_name.split("_echo-")[0] + "_desc-sepia_Chimap.nii.gz")
    os.makedirs(os.path.dirname(final_4d_chimap_file), exist_ok=True)

    for volume_input in volume_inputs:
        volume_idx = int(volume_input['volume_idx'])
        volume_label = f'vol{volume_idx:04d}'
        sepia_dir = os.path.join(temp_dir, 'sepia_matlab', volume_label)
        os.makedirs(sepia_dir, exist_ok=True)
        sepia_prefix = os.path.join(sepia_dir, 'sepia')

        base_name = os.path.basename(name_source)
        final_chimap_file = os.path.join(out_dir, base_name.split("_echo-")[0] + f"_desc-{volume_label}sepia_Chimap.nii.gz")
        os.makedirs(os.path.dirname(final_chimap_file), exist_ok=True)

        modified_sepia_script = (
            base_sepia_script.replace('{{ phase_file }}', str(volume_input['phase_file']))
            .replace('{{ mag_file }}', str(volume_input['mag_file']))
            .replace('{{ output_dir }}', sepia_prefix)
            .replace('{{ header_file }}', str(volume_input['header_file']))
            # XXX: Use mask from wk-phase-unwrap
            .replace('{{ mask_file_literal }}', _matlab_literal_path(str(run_data['mask'])))
            .replace('{{ final_chimap_file_literal }}', _matlab_literal_path(final_chimap_file))
        )

        out_sepia_script = os.path.join(sepia_dir, f'process_qsm_sepia_{volume_label}.m')
        with open(out_sepia_script, 'w') as fobj:
            fobj.write(modified_sepia_script)
        script_paths.append(out_sepia_script)
        chimap_files.append(final_chimap_file)

    concat_spec = {
        'inputs': chimap_files,
        'output': final_4d_chimap_file,
    }
    return script_paths, concat_spec


def parse_slurm_job_id(sbatch_stdout):
    """Extract a SLURM job id from sbatch output."""
    if not sbatch_stdout:
        return None
    match = re.search(r'Submitted batch job\s+(\d+)', sbatch_stdout)
    if not match:
        return None
    return match.group(1)


def submit_slurm_array(
    script_paths: list[str],
    temp_dir: str,
    max_concurrent: int,
    matlab_module: str,
    time_limit: str,
    memory: str,
    dry_run: bool,
):
    """Submit MATLAB scripts as a SLURM array."""
    if not script_paths:
        return None

    slurm_dir = os.path.join(temp_dir, 'slurm')
    os.makedirs(slurm_dir, exist_ok=True)
    script_list = os.path.join(slurm_dir, 'sepia_matlab_scripts.txt')
    with open(script_list, 'w') as fobj:
        fobj.write('\n'.join(script_paths))
        fobj.write('\n')

    module_line = f'module load {matlab_module}' if matlab_module else ':'
    array_script = os.path.join(slurm_dir, 'run_sepia_array.sbatch')
    with open(array_script, 'w') as fobj:
        fobj.write(
            '#!/bin/bash\n'
            '#SBATCH --job-name=sepia-qsm\n'
            f'#SBATCH --output={slurm_dir}/sepia-%A_%a.out\n'
            f'#SBATCH --error={slurm_dir}/sepia-%A_%a.err\n'
            f'#SBATCH --array=1-{len(script_paths)}%{max_concurrent}\n'
            '#SBATCH --cpus-per-task=1\n'
            f'#SBATCH --time={time_limit}\n'
            f'#SBATCH --mem={memory}\n'
            '\n'
            'set -euo pipefail\n'
            f'{module_line}\n'
            f'SCRIPT_LIST="{script_list}"\n'
            'SCRIPT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${SCRIPT_LIST}")\n'
            'matlab -nodisplay -nosplash -nodesktop -r "run(\'${SCRIPT}\'); exit;"\n'
        )

    if dry_run:
        print(f'Dry run: wrote SLURM script {array_script}', flush=True)
        return None

    result = subprocess.run(['sbatch', array_script], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f'sbatch failed with code {result.returncode}\n'
            f'stdout:\n{result.stdout}\n'
            f'stderr:\n{result.stderr}'
        )
    print(result.stdout.strip(), flush=True)
    return result.stdout.strip()


def write_concat_script(concat_specs: list[dict], temp_dir: str):
    """Write the dependent Python script that concatenates volume-wise chi maps."""
    if not concat_specs:
        return None

    slurm_dir = os.path.join(temp_dir, 'slurm')
    os.makedirs(slurm_dir, exist_ok=True)
    manifest_file = os.path.join(slurm_dir, 'concat_chimap_manifest.json')
    with open(manifest_file, 'w') as fobj:
        json.dump(concat_specs, fobj, indent=2)

    concat_script = os.path.join(slurm_dir, 'concat_chimaps.py')
    with open(concat_script, 'w') as fobj:
        fobj.write(
            'import json\n'
            'import os\n'
            '\n'
            'import nibabel as nb\n'
            '\n'
            f'MANIFEST = {manifest_file!r}\n'
            '\n'
            'with open(MANIFEST) as fobj:\n'
            '    specs = json.load(fobj)\n'
            '\n'
            'for spec in specs:\n'
            '    missing = [path for path in spec["inputs"] if not os.path.isfile(path)]\n'
            '    if missing:\n'
            '        raise FileNotFoundError("Missing volume-wise Chimap files: " + repr(missing))\n'
            '    imgs = [nb.load(path) for path in spec["inputs"]]\n'
            '    out_img = nb.concat_images(imgs)\n'
            '    os.makedirs(os.path.dirname(spec["output"]), exist_ok=True)\n'
            '    out_img.to_filename(spec["output"])\n'
            '    print(f"Wrote {spec[\'output\']}")\n'
        )
    return concat_script


def submit_concat_job(
    concat_script,
    temp_dir: str,
    dependency_job_id,
    time_limit: str,
    memory: str,
    dry_run: bool,
):
    """Submit a dependent SLURM job to concatenate volume-wise chi maps."""
    if concat_script is None:
        return None

    slurm_dir = os.path.join(temp_dir, 'slurm')
    concat_sbatch = os.path.join(slurm_dir, 'concat_chimaps.sbatch')
    dependency_line = (
        f'#SBATCH --dependency=afterok:{dependency_job_id}\n' if dependency_job_id else ''
    )
    with open(concat_sbatch, 'w') as fobj:
        fobj.write(
            '#!/bin/bash\n'
            '#SBATCH --job-name=sepia-concat\n'
            f'#SBATCH --output={slurm_dir}/concat-%j.out\n'
            f'#SBATCH --error={slurm_dir}/concat-%j.err\n'
            f'{dependency_line}'
            '#SBATCH --cpus-per-task=1\n'
            f'#SBATCH --time={time_limit}\n'
            f'#SBATCH --mem={memory}\n'
            '\n'
            'set -euo pipefail\n'
            f'{shlex.quote(sys.executable)} {shlex.quote(concat_script)}\n'
        )

    if dry_run:
        print(f'Dry run: wrote SLURM script {concat_sbatch}', flush=True)
        return None

    result = subprocess.run(['sbatch', concat_sbatch], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f'sbatch failed with code {result.returncode}\n'
            f'stdout:\n{result.stdout}\n'
            f'stderr:\n{result.stderr}'
        )
    print(result.stdout.strip(), flush=True)
    return result.stdout.strip()


def process_run(
    layout: BIDSLayout,
    run_data: dict,
    out_dir: str,
    temp_dir: str,
    overwrite_unwrap: bool,
    wk_n_cpus,
    wk_extra_args: str,
):
    """Prepare one run of ME-EPI data and return MATLAB script paths."""
    header_file = os.path.join(CODE_DIR, 'processing', 'sepia_header.mat')
    header_struct = loadmat(header_file)
    header_struct['B0_dir'] = header_struct['B0_dir'].astype(float)
    header_struct['B0'] = header_struct['B0'].astype(float)
    echo_times = get_echo_times(layout, run_data['bold_phase'], header_struct)

    unwrapped_phase_files = unwrap_phase_data(
        run_data=run_data,
        temp_dir=temp_dir,
        echo_times=echo_times,
        overwrite=overwrite_unwrap,
        n_cpus=wk_n_cpus,
        extra_args=wk_extra_args,
    )
    volume_inputs = write_volume_inputs(
        run_data=run_data,
        unwrapped_phase_files=unwrapped_phase_files,
        temp_dir=temp_dir,
        header_struct=header_struct,
        echo_times=echo_times,
    )
    return write_matlab_scripts(layout, run_data, volume_inputs, out_dir, temp_dir)


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--subject-id',
        type=lambda label: label.removeprefix('sub-'),
        required=True,
    )
    parser.add_argument('--overwrite-unwrap', action='store_true')
    parser.add_argument('--wk-n-cpus', type=int, default=1)
    parser.add_argument(
        '--wk-extra-args',
        default='',
        help='Extra arguments forwarded to wk-unwrap-phase, e.g. "--wrap-limit".',
    )
    parser.add_argument('--max-concurrent', type=int, default=10)
    parser.add_argument('--matlab-module', default='matlab/R2020B')
    parser.add_argument('--time-limit', default='04:00:00')
    parser.add_argument('--memory', default='16G')
    parser.add_argument('--dry-run', action='store_true')
    return parser


def _main(argv=None):
    options = _get_parser().parse_args(argv)
    main(**vars(options))


def main(
    subject_id,
    overwrite_unwrap,
    wk_n_cpus,
    wk_extra_args,
    max_concurrent,
    matlab_module,
    time_limit,
    memory,
    dry_run,
):
    in_dir = CFG['bids_dir']
    out_dir = CFG['derivatives']['meepi']
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = os.path.join(CFG['work_dir'], 'meepi', f'sub-{subject_id}')
    os.makedirs(temp_dir, exist_ok=True)

    layout = BIDSLayout(
        in_dir,
        config=['bids'],
        validate=False,
    )

    print(f'Processing subject {subject_id}', flush=True)
    all_script_paths = []
    concat_specs = []
    sessions = _as_list(layout.get_sessions(subject=subject_id, suffix='bold'))
    if not sessions:
        sessions = [None]
    for session in sessions:
        print(f'Processing session {session}', flush=True)
        seed_query = {
            'subject': subject_id,
            'datatype': 'func',
            'echo': 1,
            'part': 'mag',
            'suffix': 'bold',
            'extension': ['.nii', '.nii.gz'],
        }
        if session is not None:
            seed_query['session'] = session
        seed_files = layout.get(**seed_query)
        for seed_file in seed_files:
            entities = seed_file.get_entities()
            entities.pop('echo', None)
            entities.pop('part', None)
            entities.pop('extension', None)
            try:
                run_data = collect_run_data(layout, entities)
            except ValueError as e:
                print(f'Failed {seed_file.path}', flush=True)
                print(e, flush=True)
                continue

            fname = os.path.basename(seed_file.path).split('.')[0]
            run_temp_dir = os.path.join(temp_dir, fname.replace('-', '').replace('_', ''))
            os.makedirs(run_temp_dir, exist_ok=True)
            script_paths, concat_spec = process_run(
                layout=layout,
                run_data=run_data,
                out_dir=out_dir,
                temp_dir=run_temp_dir,
                overwrite_unwrap=overwrite_unwrap,
                wk_n_cpus=wk_n_cpus,
                wk_extra_args=wk_extra_args,
            )
            all_script_paths.extend(script_paths)
            concat_specs.append(concat_spec)

    matlab_sbatch_stdout = submit_slurm_array(
        script_paths=all_script_paths,
        temp_dir=temp_dir,
        max_concurrent=max_concurrent,
        matlab_module=matlab_module,
        time_limit=time_limit,
        memory=memory,
        dry_run=dry_run,
    )
    concat_script = write_concat_script(concat_specs, temp_dir)
    submit_concat_job(
        concat_script=concat_script,
        temp_dir=temp_dir,
        dependency_job_id=parse_slurm_job_id(matlab_sbatch_stdout),
        time_limit='01:00:00',
        memory=memory,
        dry_run=dry_run,
    )
    print('DONE!', flush=True)


if __name__ == '__main__':
    _main()
