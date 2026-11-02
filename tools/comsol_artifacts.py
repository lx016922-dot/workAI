"""Read plain CSV or an explicitly declared lossless CSV transport container.

The research server has a fixed allowlist of artifact names. A container is itself
a CSV with encoding/length/hash/payload columns; it never masquerades as numeric
waveform rows. The decoded ordinary CSV is retained alongside the signed original.
"""
import base64
import gzip
import hashlib
import io
from pathlib import Path


def properties(path):
    result={}
    for line in Path(path).read_text().splitlines():
        if line and not line.startswith(('#','!')):
            k,v=line.split('=',1)
            result[k]=v.replace('\\:',':')
    return result


def numeric_csv(path):
    path=Path(path)
    with path.open('rb') as stream:
        header=stream.readline()
    if header.rstrip()!=b'encoding,decoded_bytes,decoded_sha256,payload_base64':
        return path
    encoded=path.read_bytes().splitlines()
    if len(encoded)!=2:
        raise ValueError('Malformed CSV transport container')
    kind,count,digest,payload=encoded[1].split(b',',3)
    count=int(count);digest=digest.decode('ascii')
    receipt=properties(path.parent/'completed.properties')
    if (kind!=b'workai_gzip_csv_v1' or not 0<=count<=256*1024**2 or
        receipt.get(path.name+'_encoding')!='workai_gzip_csv_v1' or
        int(receipt.get(path.name+'_decoded_bytes',-1))!=count or
        receipt.get(path.name+'_decoded_sha256')!=digest):
        raise ValueError('Compressed artifact differs from COMSOL completion receipt')
    decoded=path.with_name(path.stem+'.decoded.csv')
    if decoded.exists():
        raw=decoded.read_bytes()
    else:
        compressed=base64.b64decode(payload,validate=True)
        with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
            raw=stream.read(count+1)
    if len(raw)!=count or hashlib.sha256(raw).hexdigest()!=digest:
        raise ValueError('Decoded CSV length/hash mismatch')
    if not decoded.exists():
        decoded.write_bytes(raw)
    return decoded
