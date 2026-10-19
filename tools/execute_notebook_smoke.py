"""Execute the standalone notebook with this interpreter in an isolated directory."""
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

root=Path(__file__).resolve().parents[1]
scratch=root/'validation_runs/notebook_execution'
scratch.mkdir(parents=True,exist_ok=True)
nb=nbformat.read(root/'solar_filament_unet_pq.ipynb',as_version=4)
manager=KernelManager(kernel_name='python3')
manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
client=NotebookClient(nb,km=manager,timeout=900,resources={'metadata':{'path':str(scratch)}})
try:
    client.execute()
finally:
    nbformat.write(nb,root/'validation_runs/executed_smoke.ipynb')
print('All standalone notebook cells executed successfully in',scratch)
