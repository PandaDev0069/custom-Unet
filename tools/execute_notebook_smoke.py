"""Execute the standalone notebook with this interpreter in an isolated directory."""
import sys
import re
import argparse
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--scratch',type=Path,default=root/'validation_runs/notebook_execution')
parser.add_argument('--output',type=Path,default=root/'validation_runs/executed_smoke.ipynb')
parser.add_argument('--workers',type=int,help='Override CPU_WORKERS for this execution only')
args=parser.parse_args()
scratch=args.scratch.resolve()
scratch.mkdir(parents=True,exist_ok=True)
nb=nbformat.read(root/'solar_filament_unet_pq.ipynb',as_version=4)
for cell in nb.cells:
    if cell.cell_type=='code' and re.search(r'^MODE\s*=',cell.source,re.M):
        cell.source=re.sub(r'^MODE\s*=.*$', "MODE = 'smoke'", cell.source, count=1, flags=re.M)
        if args.workers is not None:
            cell.source=re.sub(r'^CPU_WORKERS\s*=.*$', f'CPU_WORKERS = {args.workers}', cell.source, count=1, flags=re.M)
manager=KernelManager(kernel_name='python3')
manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
client=NotebookClient(nb,km=manager,timeout=900,resources={'metadata':{'path':str(scratch)}})
try:
    client.execute()
finally:
    args.output.parent.mkdir(parents=True,exist_ok=True)
    nbformat.write(nb,args.output)
print('All standalone notebook cells executed successfully in',scratch)
