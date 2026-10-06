"""Execução ordenada; --inprocess usa kernel IPython novo por processo sem sockets."""
from pathlib import Path
import sys, subprocess, os
import nbformat
ROOT=Path(__file__).resolve().parents[1]
def execute_inprocess(path):
    from ipykernel.inprocess.manager import InProcessKernelManager
    os.chdir(ROOT)
    nb=nbformat.read(path,as_version=4)
    manager=InProcessKernelManager(); manager.start_kernel()
    client=manager.client(); client.start_channels()
    manager.kernel.shell.run_line_magic('matplotlib','inline')
    try:
        for cell in nb.cells:
            if cell.cell_type!='code': continue
            cell.outputs=[]
            client.execute(cell.source)
            reply=client.get_shell_msg()['content']; cell.execution_count=reply['execution_count']
            while client.iopub_channel.msg_ready():
                message=client.get_iopub_msg(); kind=message['msg_type']; content=message['content']
                if kind in ['stream','display_data','execute_result','error']:
                    allowed={'stream':['name','text'],'display_data':['data','metadata'],'execute_result':['data','metadata','execution_count'],'error':['ename','evalue','traceback']}[kind]
                    cell.outputs.append(nbformat.v4.new_output(kind,**{key:content[key] for key in allowed}))
            nbformat.write(nb,path)
            if reply['status']!='ok': raise RuntimeError(f"{path.name}: {reply.get('evalue')}")
    finally:
        client.stop_channels(); manager.shutdown_kernel(); nbformat.write(nb,path)
if '--one' in sys.argv:
    execute_inprocess(Path(sys.argv[-1]))
else:
    for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
        print('Executando',path.name,flush=True)
        if '--inprocess' in sys.argv:
            subprocess.run([sys.executable,__file__,'--one',str(path)],check=True)
        else:
            from nbclient import NotebookClient
            nb=nbformat.read(path,as_version=4)
            try: NotebookClient(nb,timeout=1800,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
            finally: nbformat.write(nb,path)
        print('Concluído',path.name,flush=True)
