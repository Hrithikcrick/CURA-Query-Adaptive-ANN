"""Inventory the entire workspace and inspect local retrieval-output formats.

No pickle is executed, no index is loaded, and no retrieval is run. Git blobs
are checked read-only, including unreachable objects. All results stay local.
"""
from pathlib import Path
from collections import Counter
import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import pickletools
import re
import subprocess
import tarfile
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'sigir_analysis'
BASE=ROOT/'msmarco_large/msmarco_1000000_q1000'
QUERY_ROWS=list(csv.DictReader((BASE/'results/queries.csv').open(encoding='utf-8-sig')))
QUERIES={r['raw_query_id'] for r in QUERY_ROWS if r['source']=='trec-dl-2019'}
assert len(QUERIES)==43
ID_FIELD=re.compile(r'doc(?:ument)?[ _-]?ids?|passage[ _-]?ids?|pids?|indices|(?:^|_)rank(?:s|ed|ing|ings)?(?:_|$)|neighbou?r|retrieved|candidates?|topk[ _-]?ids|top.?\d+.*(?:ids|docs|results)|(?:exact|approx).*results|human_evidence_doc_ids_or_ranks',re.I)
CACHE_SUFFIXES={'.npy','.npz','.faiss','.index','.hnsw','.f32','.f16','.memmap','.bin','.dat','.pkl','.pickle','.parquet','.pq','.arrow','.feather','.h5','.hdf5','.pt','.pth','.joblib'}
DATA_SUFFIXES={'.trec','.run','.csv','.tsv','.json','.jsonl','.txt','.qrels'}
errors=[]


def rel(path):return path.relative_to(ROOT).as_posix()


def inspect_table(stream,delimiter):
    rows=csv.reader(stream,delimiter=delimiter)
    first=next(rows,[])
    columns=first
    matches=[c for c in columns if ID_FIELD.search(c)]
    query_columns=[i for i,c in enumerate(columns) if any(t in c.lower() for t in ['query_key','query_id','qid','raw_query'])]
    seen=set();count=0;list_fields=set();rank_rows=0
    for row in rows:
        count+=1
        for i in query_columns:
            if i<len(row):
                qid=row[i].split('::')[-1]
                if qid in QUERIES:seen.add(qid)
        for i,value in enumerate(row):
            # Lists can hide behind names other than doc IDs; record their fields.
            if value.startswith('[') and value.endswith(']'):
                list_fields.add(columns[i] if i<len(columns) else str(i))
        if matches:rank_rows+=1
    return dict(format='delimited',columns=columns,rows=count,TREC_query_ids_seen=sorted(seen),candidate_id_fields=matches,list_valued_fields=sorted(list_fields))


def inspect_json(stream,jsonl=False):
    keys=set();idkeys=set();array_paths=set();seen=set();objects=0
    def visit(value,path='',depth=0):
        nonlocal objects
        if depth>16:return
        if isinstance(value,dict):
            objects+=1
            for key,v in value.items():
                keys.add(str(key))
                if ID_FIELD.search(str(key)):idkeys.add(str(key))
                if str(key).split('::')[-1] in QUERIES:seen.add(str(key).split('::')[-1])
                visit(v,path+'/'+str(key),depth+1)
        elif isinstance(value,list):
            if len(value)>1 and all(isinstance(v,(int,str)) for v in value):array_paths.add(path)
            for v in value:visit(v,path+'[]',depth+1)
        elif isinstance(value,(str,int)) and str(value).split('::')[-1] in QUERIES:seen.add(str(value).split('::')[-1])
    if jsonl:
        for line in stream:
            if line.strip():visit(json.loads(line))
    else:visit(json.load(stream))
    return dict(format='jsonl' if jsonl else 'json',object_count=objects,keys=sorted(keys),candidate_id_fields=sorted(idkeys),scalar_array_paths=sorted(array_paths),TREC_query_ids_seen=sorted(seen))


def inspect_numpy_header(stream):
    version=np.lib.format.read_magic(stream)
    if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
    else:shape,fortran,dtype=np.lib.format.read_array_header_2_0(stream)
    return dict(shape=list(shape),dtype=str(dtype),object_dtype=bool(dtype.hasobject))


def inspect_text(stream):
    counts=Counter();seen=set();lines=0
    for line in stream:
        parts=line.split();lines+=1
        if len(parts)==4 and parts[1]=='Q0' and parts[-1].lstrip('-').isdigit():counts['qrels_rows']+=1
        if len(parts)==6 and parts[1]=='Q0' and parts[3].isdigit():counts['TREC_run_rows']+=1
        if parts and parts[0].split('::')[-1] in QUERIES:seen.add(parts[0].split('::')[-1])
    return dict(format='text',lines=lines,detected_row_formats=dict(counts),TREC_query_ids_seen=sorted(seen))


def inspect_stream(name,raw,size):
    suffix=Path(name).suffix.lower()
    if suffix=='.npy':return dict(format='npy',**inspect_numpy_header(raw))
    if suffix in {'.pkl','.pickle','.joblib'}:
        # Opcode inspection does not call arbitrary pickle reducers.
        if suffix=='.joblib':return dict(format='joblib',note='Not deserialized; candidate requires manual safe review')
        strings=[];ops=0
        for opcode,arg,pos in pickletools.genops(raw):
            ops+=1
            if isinstance(arg,str) and (ID_FIELD.search(arg) or any(q in arg for q in QUERIES)):strings.append(arg[:120])
        return dict(format='pickle_opcodes_only',opcodes=ops,candidate_strings=strings[:200])
    if suffix in {'.parquet','.pq','.arrow','.feather'}:
        import pyarrow.parquet as pq
        if suffix not in {'.parquet','.pq'}:return dict(format='arrow',note='Candidate exists; further IPC schema validation required')
        file=pq.ParquetFile(raw)
        return dict(format='parquet',columns=file.schema.names,rows=file.metadata.num_rows,candidate_id_fields=[c for c in file.schema.names if ID_FIELD.search(c)])
    if suffix in DATA_SUFFIXES:
        text=io.TextIOWrapper(raw,encoding='utf-8-sig',errors='replace')
        if suffix in {'.csv','.tsv'}:return inspect_table(text,',' if suffix=='.csv' else '\t')
        if suffix in {'.json','.jsonl'}:return inspect_json(text,suffix=='.jsonl')
        return inspect_text(text)
    return dict(format='binary_or_unknown',bytes=size)


def run(label):
    inventory=[];inspections=[];archive_members=[];containers=[]
    for directory,dirs,names in os.walk(ROOT,followlinks=False,onerror=lambda e:errors.append(str(e))):
        for name in names:
            path=Path(directory)/name
            try:
                rp=rel(path);size=path.stat().st_size;suffix=path.suffix.lower()
                auxiliary=any(p in path.parts for p in ['.plot_deps','.mplconfig','__pycache__'])
                git='.git' in path.parts
                inventory.append(dict(path=rp,bytes=size,suffix=suffix,category='git_objects_or_metadata' if git else 'analysis_runtime' if auxiliary else 'workspace'))
                if auxiliary or git:continue
                magic=path.open('rb').read(12)
                archive=suffix in {'.zip','.npz','.tar','.tgz','.gz','.bz2','.xz','.7z'}
                if zipfile.is_zipfile(path):
                    with zipfile.ZipFile(path) as z:
                        containers.append(dict(path=rp,type='npz' if suffix=='.npz' else 'zip',members=len(z.infolist())))
                        for member in z.infolist():
                            if member.is_dir():continue
                            ms=Path(member.filename).suffix.lower()
                            archive_members.append(dict(container=rp,member=member.filename,bytes=member.file_size,suffix=ms))
                            if ms in CACHE_SUFFIXES or ms in DATA_SUFFIXES:
                                with z.open(member) as stream:
                                    inspection=inspect_stream(member.filename,stream,member.file_size)
                                inspections.append(dict(path=rp+'::'+member.filename,**inspection))
                elif archive:
                    if tarfile.is_tarfile(path):
                        with tarfile.open(path) as tf:
                            containers.append(dict(path=rp,type='tar'))
                            for member in tf:
                                if not member.isfile():continue
                                archive_members.append(dict(container=rp,member=member.name,bytes=member.size,suffix=Path(member.name).suffix.lower()))
                                if Path(member.name).suffix.lower() in CACHE_SUFFIXES|DATA_SUFFIXES:
                                    with tf.extractfile(member) as stream:inspection=inspect_stream(member.name,stream,member.size)
                                    inspections.append(dict(path=rp+'::'+member.name,**inspection))
                    elif suffix=='.gz':
                        with gzip.open(path,'rb') as stream:inspection=inspect_stream(path.stem,stream,size)
                        inspections.append(dict(path=rp,**inspection))
                    else:errors.append('Unsupported archive format: '+rp)
                elif suffix in CACHE_SUFFIXES|DATA_SUFFIXES or magic.startswith(b'\x93NUMPY'):
                    with path.open('rb') as stream:inspection=inspect_stream(path.name if not magic.startswith(b'\x93NUMPY') else 'array.npy',stream,size)
                    inspections.append(dict(path=rp,**inspection))
            except Exception as exc:errors.append(dict(path=rel(path),error=str(exc),type=type(exc).__name__))
    # Read every local Git blob, so removed/unreachable caches cannot be missed
    # solely because they no longer have a working-tree filename.
    git_result=subprocess.run(['git','cat-file','--batch-all-objects','--batch-check=%(objectname) %(objecttype) %(objectsize)'],cwd=ROOT,capture_output=True,text=True)
    git_blobs=[];git_candidates=[]
    if git_result.returncode==0:
        for line in git_result.stdout.splitlines():
            oid,kind,size=line.split()
            if kind=='blob':git_blobs.append((oid,int(size)))
        for oid,size in git_blobs:
            blob=subprocess.run(['git','cat-file','blob',oid],cwd=ROOT,capture_output=True,check=True).stdout
            if blob.startswith(b'\x93NUMPY') or blob.startswith(b'PAR1') or blob.startswith(b'\x80\x04'):
                git_candidates.append(dict(oid=oid,bytes=size,kind='binary_retrieval_candidate'))
            elif blob.startswith(b'PK\x03\x04'):
                with zipfile.ZipFile(io.BytesIO(blob)) as z:
                    hits=[n for n in z.namelist() if Path(n).suffix.lower() in CACHE_SUFFIXES or re.search(r'qrels|rankings|docids',n,re.I)]
                if hits:git_candidates.append(dict(oid=oid,bytes=size,kind='archive',candidate_members=hits))
            elif size<25_000_000:
                text=blob.decode('utf-8',errors='replace')
                first=text.splitlines()[0] if text.splitlines() else ''
                if re.fullmatch(r'\d+ Q0 \S+ [-]?\d+',first.strip()) or re.fullmatch(r'\d+ Q0 \S+ \d+ \S+ \S+',first.strip()):
                    git_candidates.append(dict(oid=oid,bytes=size,kind='qrels_or_run',first_row_format='qid Q0 document label/rank'))
                if first and ID_FIELD.search(first) and (',' in first or '\t' in first):
                    fields=next(csv.reader([first],delimiter='\t' if '\t' in first else ','))
                    matches=[field for field in fields if ID_FIELD.search(field)]
                    if matches:git_candidates.append(dict(oid=oid,bytes=size,kind='tabular',candidate_id_fields=matches))
    else:errors.append(dict(git_error=git_result.stderr))
    def write_csv(name,rows,fields):
        with (OUT/name).open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    write_csv('local_artifact_inventory.csv',inventory,['path','bytes','suffix','category'])
    write_csv('archive_member_inventory.csv',archive_members,['container','member','bytes','suffix'])
    cache=[r for r in inventory if r['suffix'] in CACHE_SUFFIXES and r['category']=='workspace']
    result=dict(scope='entire CURA workspace recursively, including hidden/ignored files, archive content and all Git blobs',phase=label,file_count=len(inventory),workspace_file_count=sum(r['category']=='workspace' for r in inventory),suffix_counts=dict(Counter(r['suffix'] for r in inventory)),cache_candidates=cache,containers=containers,content_inspections=inspections,git_blob_count=len(git_blobs),git_candidates=git_candidates,errors=errors,TREC_query_count=len(QUERIES),retrieval_executed=False)
    name='local_retrieval_audit.json' if label=='current' else 'local_retrieval_audit_'+label+'.json'
    (OUT/name).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(phase=label,files=len(inventory),inspected_formats=len(inspections),archives=len(containers),git_blobs=len(git_blobs),git_candidates=len(git_candidates),cache_candidates=cache,errors=errors),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',default='current');args=parser.parse_args();run(args.phase)
