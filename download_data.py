"""Download the authors' public archive. Dataset is never committed."""
import hashlib
from pathlib import Path
import requests
from study import URL

EXPECTED = '7a5564290a504bdc104dccaf44aa6c0b3857bf1db6b0517b9b88577c2def3e5f'
def main():
    p=Path(__file__).resolve().parent/'data/data.zip'
    p.parent.mkdir(exist_ok=True)
    if p.exists():
        content=p.read_bytes()
    else:
        r=requests.get(URL,timeout=180)
        r.raise_for_status()
        content=r.content
    if hashlib.sha256(content).hexdigest()!=EXPECTED:
        raise ValueError('Dataset changed: inspect provenance before running; file not written.')
    p.write_bytes(content)
    print('Verified archive:',p)
if __name__=='__main__': main()
