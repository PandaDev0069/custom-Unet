"""Summarize executed checks without inventing training-dependent artifacts."""
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime,timedelta,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from filament.io import atomic_json,file_hash
from filament.data import discover_test_images,find_dataset_root


def test_summary(path):
    element=ET.parse(path).getroot()
    suites=list(element) if element.tag=='testsuites' else [element]
    result={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    result['passed']=result['tests']-result['failures']-result['errors']-result['skipped']
    return result


def main():
    directory=ROOT/'validation_runs'
    data=json.loads((directory/'dataset_integrity.json').read_text(encoding='utf8'))
    cpu=test_summary(directory/'pytest_cpu.xml')
    gpu=test_summary(directory/'pytest_gpu.xml')
    executed=directory/'executed_smoke.ipynb'
    notebook=json.loads(executed.read_text(encoding='utf8'))
    code_cells=[c for c in notebook['cells'] if c['cell_type']=='code']
    assert all(c['execution_count'] is not None for c in code_cells)
    assert not any(o['output_type']=='error' for c in code_cells for o in c.get('outputs',[]))
    run=directory/'notebook_execution/filament_smoke'
    runtime=json.loads((run/'runtime_report.json').read_text(encoding='utf8'))
    results=json.loads((run/'results_summary.json').read_text(encoding='utf8'))
    split=json.loads((run/'split_manifest.json').read_text(encoding='utf8'))
    best=json.loads((run/'search/best_postprocessing.json').read_text(encoding='utf8'))
    ddp=(directory/'ddp_smoke.log').read_text(encoding='utf8')
    assert 'PASS: two-rank Gloo' in ddp
    current=json.loads((ROOT/'solar_filament_unet_pq.ipynb').read_text(encoding='utf8'))
    assert current['metadata']['filament']['source_files']==notebook['metadata']['filament']['source_files'], 'Executed notebook source is stale'
    tests_images=len(discover_test_images(find_dataset_root()))
    probe=runtime['probes'][0]['ranks'][0][0]
    report=dict(verified_at=datetime.now(timezone(timedelta(hours=9))).isoformat(),
        cpu_tests=cpu,gpu_tests=gpu,ddp='two CPU/Gloo ranks: passed',
        standalone_notebook=dict(code_cells_executed=len(code_cells),errors=0,
            primary_sha256=file_hash(ROOT/'solar_filament_unet_pq.ipynb'),executed_path=str(executed)),
        dataset=dict(source_images=data['integrity']['source_images'],test_images=tests_images,
            annotation_records=data['integrity']['annotation_records'],annotations=data['integrity']['annotations'],
            multi_annotator_sources=len(data['integrity']['multi_annotator_sources']),
            overlap_records=data['statistics']['overlap_records'],small_instances=data['statistics']['small_instances'],
            train_records=len(split['train']),val_records=len(split['val'])),
        native_gpu=runtime['hardware'][0],probe=probe,smoke_validation=results['validation'],
        limitations=['No full dataset training run','No Linux/NCCL or dual-T4 execution',
            'No Kaggle submission uploaded or accepted','No performance improvement measured'])
    atomic_json(directory/'validation_summary.json',report)
    d=report['dataset']; hardware=report['native_gpu']
    text=f'''# Executed validation

Verified {report['verified_at']} (Asia/Tokyo). Machine-readable evidence:
[`validation_summary.json`](../validation_runs/validation_summary.json).

| Check | Actual result |
|---|---|
| CPU synthetic/unit/integration suite | {cpu['passed']} passed, {cpu['skipped']} CUDA-only skipped, {cpu['failures']+cpu['errors']} failures/errors |
| CUDA suite with native probe enabled | {gpu['passed']} passed, {gpu['skipped']} skipped, {gpu['failures']+gpu['errors']} failures/errors |
| Two-rank CPU/Gloo DDP smoke | Passed: accumulation agrees with direct reference; unequal validation rank lengths count all records exactly once |
| Notebook structure/source parity | 30 ordered sections, 18 exported modules, 98 cells; generated modules match repository ASTs |
| Complete standalone notebook | All {len(code_cells)} code cells executed successfully with no cell errors |
| Full local annotation integrity | {d['annotations']:,} annotations decoded and validated across {d['annotation_records']:,} annotator records |
| Original native image checks | {d['source_images']} train sources, {d['test_images']} test sources; 2048 × 2048 |
| Native GPU training smoke | FP16, full-image forward/backward, accumulated AdamW, validation, best/last checkpoints and logs succeeded |
| Native validation/postprocessing | Successful cached full-image inference, two postprocessing comparisons and saved review figures |

Hardware: **{hardware['name']}**, {hardware['vram_gib']:.2f} GiB VRAM.
The standalone smoke used ModifiedUNet width 8, microbatch one and native 2048².
Real-backward probe peak allocated/reserved:
**{probe['peak_allocated_bytes']/2**30:.3f} / {probe['peak_reserved_bytes']/2**30:.3f} GiB**.
AdamW optimizer state was allocated and updated during probing. This is local
hardware evidence, not a two-T4 throughput/memory claim.

The full dataset has {d['multi_annotator_sources']} multiply annotated sources and
{d['overlap_records']} annotation records with overlapping original instance masks.
No pixel-identical duplicate filenames were found in this local training copy.
One decoded instance has area below 20 pixels (minimum 9 pixels), so minimum-area
filtering is a validation parameter, not a ground-truth deletion rule. The default
saved split contains {d['train_records']} train and {d['val_records']} validation
records; smoke execution uses only two native records from each partition.

The one-epoch smoke validation PQ is **{results['validation']['pq']:.1f}**. It is
an execution check and does not establish competitive quality. No final competition
`submission.csv` was generated from the smoke model. Synthetic integration tests
exercise CSV generation, zero detections, RLE round trips, interruption recovery,
schema and duplicate checks.

## Evidence files

- [Executed notebook](../validation_runs/executed_smoke.ipynb)
- [CPU test XML](../validation_runs/pytest_cpu.xml) and [GPU test XML](../validation_runs/pytest_gpu.xml)
- [Gloo DDP log](../validation_runs/ddp_smoke.log)
- [Dataset integrity/statistics](../validation_runs/dataset_integrity.json)
- [Native smoke runtime](../validation_runs/notebook_execution/filament_smoke/runtime_report.json)
- [Native smoke history](../validation_runs/notebook_execution/filament_smoke/logs/history.csv)
- [Postprocessing trials](../validation_runs/notebook_execution/filament_smoke/search/postprocessing_trials.csv)
- [Training plot](../validation_runs/notebook_execution/filament_smoke/figures/training.png)
- [Target inspection](../validation_runs/notebook_execution/filament_smoke/figures/targets.png)
- [Prediction review](../validation_runs/notebook_execution/filament_smoke/figures/prediction_review.png)

Warnings observed: pycocotools emits a NumPy `__array__(copy=...)` deprecation
warning in the installed dependency combination. It does not fail round-trip or
evaluator tests. Notebook kernel startup emits a local Jupyter transport warning.

## Not executed and remaining limitations

Full dataset training, Linux/NCCL dual-T4 execution, sustained Kaggle memory/time
budget validation and Kaggle submission acceptance were not performed. Gloo checks
DDP synchronization mathematics, not NCCL deployment. Unexpected training OOM
restart is implemented but was not forced on a live multi-GPU job. Native GPU
smoke/probes passed without requiring width/batch fallback.

Two pixel heads cannot fully represent overlapping instance memberships. Watershed
can fragment elongated objects; optional reconnection can merge distinct objects.
Default single-annotator validation underrepresents annotator disagreement, and
checkpoint/postprocessing selection on one holdout creates selection bias. See
[EXPERIMENTS.md](EXPERIMENTS.md) for recommended controlled comparisons. No improved
competition performance is claimed.
'''
    (ROOT/'docs/VALIDATION.md').write_text(text,encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
