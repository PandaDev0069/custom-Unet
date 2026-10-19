"""Validate section ordering, exported source parity and cell Python syntax."""
import ast
import json
import re
from pathlib import Path
import nbformat
from IPython.core.interactiveshell import InteractiveShell

ROOT=Path(__file__).resolve().parents[1]


def verify():
    notebook=nbformat.read(ROOT/'solar_filament_unet_pq.ipynb',as_version=4)
    nbformat.validate(notebook)
    sections=[]; reconstructed={}
    shell=InteractiveShell.instance()
    for index,cell in enumerate(notebook.cells):
        text=cell.source
        if cell.cell_type=='markdown':
            match=re.match(r'^#{1,2} (\d+)\.',text)
            if match: sections.append(int(match[1]))
        else:
            if text.startswith('%%writefile '):
                header,source=text.split('\n',1)
                path=header.split('filament_runtime/',1)[1]
                if '-a ' in header: reconstructed[path]+=source
                else: reconstructed[path]=source
            compile(shell.transform_cell(text),f'cell_{index}','exec')
    assert sections==list(range(1,31)),sections
    for name,source in reconstructed.items():
        canonical=(ROOT/name).read_text(encoding='utf8')
        # Chunks may normalize a blank line at statement boundaries only.
        assert ast.dump(ast.parse(source))==ast.dump(ast.parse(canonical)),name
    assert all(not c.get('outputs') for c in notebook.cells if c.cell_type=='code')
    print(f'Notebook verified: 30 ordered sections, {len(reconstructed)} source modules, {len(notebook.cells)} valid cells')


if __name__=='__main__': verify()
