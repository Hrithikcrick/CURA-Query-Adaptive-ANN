"""Inventory the entire workspace and inspect local retrieval-output formats.

No pickle is executed, no index is loaded, and no retrieval is run. Git blobs
are checked read-only, including unreachable objects. All results stay local.
"""
from pathlib import Path
from collections import Counter
import argparse
import bz2
import csv
import gzip
import hashlib
import io
import json
import lzma
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
TEXT_SUFFIXES=DATA_SUFFIXES|{'.py','.ipynb','.md','.tex','.rst','.yaml','.yml','.toml','.ini','.cfg','.log','.bak','.backup','.old','.orig','.ps1','.sh','.bat','.r','.m','.sql'}
ARCHIVE_SUFFIXES={'.zip','.npz','.tar','.tgz','.gz','.bz2','.xz','.7z'}
PATH_HINT=re.compile(r'''(?:[A-Za-z]:[\\/])?(?:[\w.-]{1,120}[\\/]){0,15}[\w.-]{0,100}(?:embedding|vector|docids?|rankings?|top\d+|hnsw|faiss|qrels|corpus|candidates?)[\w.-]{0,100}\.(?:npy|npz|f32|f16|faiss|index|hnsw|pkl|pickle|parquet|csv|tsv|jsonl?|trec|run|txt|bin|dat)''',re.I)
DOC_FIELD=re.compile(r'^(?:doc(?:ument)?_?ids?|passage_?ids?|pids?|retrieved_ids|topk_ids|candidate_ids|indices)$',re.I)
RANK_FIELD=re.compile(r'^(?:rank|ranks|position|positions)$',re.I)
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
    plausible=len(seen)==43 and any(DOC_FIELD.fullmatch(c) for c in columns) and (bool(list_fields) or any(RANK_FIELD.fullmatch(c) for c in columns))
    return dict(format='delimited',columns=columns,rows=count,TREC_query_ids_seen=sorted(seen),candidate_id_fields=matches,list_valued_fields=sorted(list_fields),plausible_complete_ranking_candidate=plausible)


def inspect_json(stream,jsonl=False):
    keys=set();idkeys=set();array_paths=set();seen=set();objects=0;rank_arrays=[];rank_records=[];notebook_text=[]
    def visit(value,path='',depth=0):
        nonlocal objects
        if isinstance(value,dict):
            objects+=1
            # A metric called "rankings_recovered" is not a document ranking.
            # Require nonempty ID arrays or actual query/document/rank records.
            if any(DOC_FIELD.fullmatch(str(k)) and isinstance(v,(str,int)) and str(v).strip() for k,v in value.items()) and any(RANK_FIELD.fullmatch(str(k)) for k in value):
                rank_records.append(path)
            for key,v in value.items():
                keys.add(str(key))
                if ID_FIELD.search(str(key)):idkeys.add(str(key))
                if str(key).split('::')[-1] in QUERIES:seen.add(str(key).split('::')[-1])
                if str(key) in {'text','source','data','text/plain','text/html','application/json'} and isinstance(v,(str,list)):
                    if isinstance(v,str):notebook_text.append(v)
                    elif all(isinstance(x,str) for x in v):notebook_text.append(''.join(v))
                visit(v,path+'/'+str(key),depth+1)
        elif isinstance(value,list):
            if len(value)>1 and all(isinstance(v,(int,str)) and not isinstance(v,bool) for v in value):
                array_paths.add(path)
                field=path.rsplit('/',1)[-1]
                if DOC_FIELD.fullmatch(field) or field.split('::')[-1] in QUERIES:
                    rank_arrays.append(dict(path=path,items=len(value)))
            for v in value:visit(v,path+'[]',depth+1)
        elif isinstance(value,(str,int)) and str(value).split('::')[-1] in QUERIES:seen.add(str(value).split('::')[-1])
    if jsonl:
        for line in stream:
            if line.strip():visit(json.loads(line))
    else:visit(json.load(stream))
    result=dict(format='jsonl' if jsonl else 'json',object_count=objects,keys=sorted(keys),candidate_id_fields=sorted(idkeys),scalar_array_paths=sorted(array_paths),TREC_query_ids_seen=sorted(seen),ranking_evidence=dict(document_id_arrays=rank_arrays,document_rank_record_paths=rank_records),plausible_complete_ranking_candidate=len(seen)==43 and bool(rank_arrays or rank_records))
    if 'cells' in keys:
        result['format']='notebook'
        # Include notebook source, stream outputs, text/plain and HTML display
        # outputs. Embedded JSON is also traversed by visit above.
        embedded=[]
        for index,text in enumerate(notebook_text):
            detail=inspect_unlabelled_text(text)
            if detail.get('path_hints') or detail.get('detected_row_formats') or detail.get('embedded_delimited_tables') or detail.get('ranking_evidence') or detail.get('candidate_id_fields'):
                embedded.append(dict(output_index=index,**detail))
        result['notebook_source_and_output_inspections']=embedded
        result['plausible_complete_ranking_candidate'] |= any(d.get('plausible_complete_ranking_candidate',False) for d in embedded)
    return result


def inspect_numpy_header(stream):
    version=np.lib.format.read_magic(stream)
    if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
    else:shape,fortran,dtype=np.lib.format.read_array_header_2_0(stream)
    return dict(shape=list(shape),dtype=str(dtype),object_dtype=bool(dtype.hasobject))


def inspect_text(stream):
    counts=Counter();seen=set();lines=0;hints=set();tables=[];current=[];current_delimiter=None
    def finish_table():
        nonlocal current,current_delimiter
        if current:
            result=inspect_table(io.StringIO('\n'.join(current)),current_delimiter)
            if result['candidate_id_fields']:tables.append(result)
        current=[];current_delimiter=None
    for line in stream:
        parts=line.split();lines+=1
        if len(parts)==4 and parts[1] in {'Q0','0'} and parts[-1].lstrip('-').isdigit():counts['qrels_rows']+=1
        if len(parts)==6 and parts[1]=='Q0' and parts[3].isdigit():counts['TREC_run_rows']+=1
        if parts and parts[0].split('::')[-1] in QUERIES:seen.add(parts[0].split('::')[-1])
        hints.update(m.group().strip() for m in PATH_HINT.finditer(line))
        # Displayed notebook CSV/TSV rankings need schema/content inspection,
        # including when the output has no .csv filename.
        delimiter='\t' if '\t' in line else ',' if ',' in line else None
        if current and delimiter==current_delimiter:current.append(line.rstrip('\r\n'))
        elif delimiter and ID_FIELD.search(line) and any(t in line.lower() for t in ['qid','query_id','query_key','raw_query']):
            finish_table();current=[line.rstrip('\r\n')];current_delimiter=delimiter
        else:finish_table()
    finish_table()
    plausible=bool(counts['TREC_run_rows'] and len(seen)==43) or any(len(t['TREC_query_ids_seen'])==43 and any(DOC_FIELD.fullmatch(c) for c in t['columns']) and any(RANK_FIELD.fullmatch(c) for c in t['columns']) for t in tables)
    return dict(format='text',lines=lines,detected_row_formats=dict(counts),TREC_query_ids_seen=sorted(seen),path_hints=sorted(hints),embedded_delimited_tables=tables,plausible_complete_ranking_candidate=plausible)


def inspect_stream(name,raw,size):
    suffix=Path(name).suffix.lower()
    # Detect formats by bytes as well as names, covering backups and nameless
    # Git blobs. A buffered in-memory prefix does not execute a serialization.
    prefix=raw.read(8)
    raw=io.BufferedReader(PrefixReader(prefix,raw))
    if prefix.startswith(b'\x93NUMPY'):suffix='.npy'
    elif prefix.startswith(b'PAR1'):suffix='.parquet'
    elif len(prefix)>1 and prefix[0]==128 and prefix[1] in range(0,6):suffix='.pickle'
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
    if suffix in TEXT_SUFFIXES or suffix in {'.json','.jsonl'}:
        text=io.TextIOWrapper(raw,encoding='utf-8-sig',errors='replace')
        if suffix in {'.csv','.tsv'}:return inspect_table(text,',' if suffix=='.csv' else '\t')
        if suffix in {'.json','.jsonl','.ipynb'}:return inspect_json(text,suffix=='.jsonl')
        content=text.read()
        # Backup extensions often conceal the original JSON/CSV filename.
        return inspect_unlabelled_text(content)
    return dict(format='binary_or_unknown',bytes=size)


class PrefixReader(io.RawIOBase):
    """Reattach a sniffed prefix without seeking compressed archive streams."""
    def __init__(self,prefix,stream):self.prefix=io.BytesIO(prefix);self.stream=stream
    def readable(self):return True
    def readinto(self,target):
        data=self.prefix.read(len(target))
        if len(data)<len(target):data+=self.stream.read(len(target)-len(data))
        target[:len(data)]=data
        return len(data)


def inspect_unlabelled_text(content):
    stripped=content.lstrip('\ufeff \r\n\t')
    if stripped.startswith(('{','[')):
        try:return inspect_json(io.StringIO(content))
        except json.JSONDecodeError:pass
        # JSONL need not have a .jsonl filename.
        try:return inspect_json(io.StringIO(content),True)
        except json.JSONDecodeError:pass
    first=content.splitlines()[0] if content.splitlines() else ''
    if first and ID_FIELD.search(first) and (',' in first or '\t' in first):
        delimiter='\t' if '\t' in first else ','
        fields=next(csv.reader([first],delimiter=delimiter))
        if any(ID_FIELD.search(f) for f in fields) and not first.lstrip().startswith(('#','\'')):
            result=inspect_table(io.StringIO(content),delimiter)
            result['path_hints']=sorted({m.group().strip() for m in PATH_HINT.finditer(content)})
            return result
    return inspect_text(io.StringIO(content))


def inspect_payload(name,raw,size,path,inspections,archive_members,containers,depth=0):
    """Read nested containers without extracting files or invoking loaders."""
    suffix=Path(name).suffix.lower()
    prefix=raw.read(512);raw.seek(0)
    zipped=prefix.startswith((b'PK\x03\x04',b'PK\x05\x06',b'PK\x07\x08'))
    if zipped:
        with zipfile.ZipFile(raw) as archive:
            containers.append(dict(path=path,type='npz' if suffix=='.npz' else 'zip',members=len(archive.infolist())))
            for member in archive.infolist():
                if member.is_dir():continue
                ms=Path(member.filename).suffix.lower()
                archive_members.append(dict(container=path,member=member.filename,bytes=member.file_size,suffix=ms))
                with archive.open(member) as stream:
                    inspect_payload(member.filename,stream,member.file_size,path+'::'+member.filename,inspections,archive_members,containers,depth+1)
        return
    if prefix.startswith(b'\x1f\x8b') or suffix in {'.gz','.tgz'}:
        with gzip.GzipFile(fileobj=raw) as stream:
            inspect_payload('payload.tar' if suffix=='.tgz' else Path(name).stem,stream,size,path+'::decompressed',inspections,archive_members,containers,depth+1)
        return
    if prefix.startswith(b'BZh') or suffix=='.bz2':
        with bz2.BZ2File(raw) as stream:inspect_payload(Path(name).stem,stream,size,path+'::decompressed',inspections,archive_members,containers,depth+1)
        return
    if prefix.startswith(b'\xfd7zXZ\x00') or suffix=='.xz':
        with lzma.LZMAFile(raw) as stream:inspect_payload(Path(name).stem,stream,size,path+'::decompressed',inspections,archive_members,containers,depth+1)
        return
    if prefix[257:262]==b'ustar' or suffix=='.tar':
        with tarfile.open(fileobj=raw,mode='r:') as archive:
            containers.append(dict(path=path,type='tar'))
            for member in archive:
                if not member.isfile():continue
                ms=Path(member.name).suffix.lower()
                archive_members.append(dict(container=path,member=member.name,bytes=member.size,suffix=ms))
                with archive.extractfile(member) as stream:inspect_payload(member.name,stream,member.size,path+'::'+member.name,inspections,archive_members,containers,depth+1)
        return
    if suffix=='.7z':
        errors.append(dict(path=path,error='7z archive requires a safe reader unavailable to this audit',type='UnsupportedArchive'))
        return
    if suffix in CACHE_SUFFIXES|TEXT_SUFFIXES or prefix.startswith((b'\x93NUMPY',b'PAR1')) or (len(prefix)>1 and prefix[0]==128 and prefix[1] in range(0,6)):
        inspections.append(dict(path=path,**inspect_stream(name,raw,size)))
    elif not prefix or b'\x00' not in prefix:
        # Nameless Git blobs and uncommon backup extensions can contain JSON,
        # plain TREC runs, source code, or a tabular retrieval output.
        content=raw.read().decode('utf-8-sig',errors='replace')
        inspections.append(dict(path=path,**inspect_unlabelled_text(content)))
    else:inspections.append(dict(path=path,format='binary_or_unknown',bytes=size,binary_signature='PNG' if prefix.startswith(b'\x89PNG\r\n\x1a\n') else None,note='Safely inventoried; not deserialized or treated as recovered retrieval state'))


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
                with path.open('rb') as stream:inspect_payload(path.name,stream,size,rp,inspections,archive_members,containers)
            except Exception as exc:errors.append(dict(path=rel(path),error=str(exc),type=type(exc).__name__))
    # Read every local Git blob, so removed/unreachable caches cannot be missed
    # solely because they no longer have a working-tree filename.
    git_result=subprocess.run(['git','cat-file','--batch-all-objects','--batch-check=%(objectname) %(objecttype) %(objectsize)'],cwd=ROOT,capture_output=True,text=True)
    git_blobs=[];git_candidates=[];git_inspections=[];git_containers=[];git_members=[]
    # Map both current history and reflog-reachable paths, but enumerate all
    # object-store blobs independently so deleted/unreachable blobs are read too.
    path_result=subprocess.run(['git','rev-list','--objects','--all','--reflog'],cwd=ROOT,capture_output=True,text=True)
    git_paths={}
    if path_result.returncode==0:
        for line in path_result.stdout.splitlines():
            oid,_,name=line.partition(' ')
            if name:git_paths.setdefault(oid,[]).append(name)
    else:errors.append(dict(git_path_history_error=path_result.stderr))
    ref_result=subprocess.run(['git','for-each-ref','--format=%(refname) %(objectname)'],cwd=ROOT,capture_output=True,text=True)
    reflog_result=subprocess.run(['git','reflog','show','--all','--format=%H'],cwd=ROOT,capture_output=True,text=True)
    unreachable_result=subprocess.run(['git','fsck','--full','--no-reflogs','--unreachable','--no-progress'],cwd=ROOT,capture_output=True,text=True)
    unreachable_blobs={line.split()[-1] for line in unreachable_result.stdout.splitlines() if line.startswith('unreachable blob ')}
    if git_result.returncode==0:
        for line in git_result.stdout.splitlines():
            oid,kind,size=line.split()
            if kind=='blob':git_blobs.append((oid,int(size)))
        for oid,size in git_blobs:
            blob=subprocess.run(['git','cat-file','blob',oid],cwd=ROOT,capture_output=True,check=True).stdout
            try:
                paths=git_paths.get(oid,[]);name=paths[0] if paths else 'unnamed_git_blob'
                begin=len(git_inspections);before_members=len(git_members)
                inspect_payload(name,io.BytesIO(blob),size,'git:'+oid,git_inspections,git_members,git_containers)
                details=git_inspections[begin:]
                for detail in details:
                    detail.update(oid=oid,history_paths=paths,unreachable_without_reflogs=oid in unreachable_blobs)
                    kind=detail['format'];fields=detail.get('candidate_id_fields',[])
                    # Retain exact tabular semantics for the existing verifier.
                    if kind=='delimited' and fields:
                        git_candidates.append(dict(oid=oid,bytes=size,kind='tabular' if detail['path']=='git:'+oid else 'archive_member_tabular',candidate_id_fields=fields,inspection_path=detail['path'],TREC_query_ids_seen=detail.get('TREC_query_ids_seen',[]),plausible_complete_ranking_candidate=detail.get('plausible_complete_ranking_candidate',False)))
                    elif kind in {'json','jsonl','notebook'} and (fields or detail.get('ranking_evidence',{}).get('document_id_arrays') or detail.get('ranking_evidence',{}).get('document_rank_record_paths')):
                        git_candidates.append(dict(oid=oid,bytes=size,kind=kind,inspection_path=detail['path'],TREC_query_ids_seen=detail.get('TREC_query_ids_seen',[]),ranking_evidence=detail.get('ranking_evidence',{}),plausible_complete_ranking_candidate=detail.get('plausible_complete_ranking_candidate',False),note='Scalar metric keys alone are not accepted ranking evidence'))
                    elif kind=='text' and detail.get('detected_row_formats'):
                        git_candidates.append(dict(oid=oid,bytes=size,kind='qrels_or_run',inspection_path=detail['path'],detected_row_formats=detail['detected_row_formats'],TREC_query_ids_seen=detail.get('TREC_query_ids_seen',[]),plausible_complete_ranking_candidate=detail.get('plausible_complete_ranking_candidate',False)))
                    elif kind in {'npy','parquet','pickle_opcodes_only','joblib','arrow'}:
                        git_candidates.append(dict(oid=oid,bytes=size,kind='binary_retrieval_candidate',inspection_path=detail['path'],details=detail))
                if len(git_members)>before_members:
                    hits=[m['member'] for m in git_members[before_members:] if m['suffix'] in CACHE_SUFFIXES or re.search(r'qrels|rankings|docids',m['member'],re.I)]
                    if hits:git_candidates.append(dict(oid=oid,bytes=size,kind='archive',candidate_members=hits))
            except Exception as exc:errors.append(dict(git_blob=oid,error=str(exc),type=type(exc).__name__))
    else:errors.append(dict(git_error=git_result.stderr))
    def write_csv(name,rows,fields):
        with (OUT/name).open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    write_csv('local_artifact_inventory.csv',inventory,['path','bytes','suffix','category'])
    write_csv('archive_member_inventory.csv',archive_members+git_members,['container','member','bytes','suffix'])
    cache=[r for r in inventory if r['suffix'] in CACHE_SUFFIXES and r['category']=='workspace']
    unresolved=[dict(path=r['path'],format=r['format'],reason='Potential full 43-query ranking structure requires provenance and metric reproduction') for r in inspections+git_inspections if r.get('plausible_complete_ranking_candidate')]
    hints=[dict(path=r['path'],path_hints=r['path_hints']) for r in inspections+git_inspections if r.get('path_hints')]
    git_scope=dict(object_enumeration='git cat-file --batch-all-objects; all blobs read without the former 25 MB text limit',history_paths='git rev-list --objects --all --reflog',refs=ref_result.stdout.splitlines() if ref_result.returncode==0 else [],reflog_entries=len(reflog_result.stdout.splitlines()) if reflog_result.returncode==0 else None,unreachable_blob_count_without_reflogs=len(unreachable_blobs),unreachable_blobs_inspected=sum(oid in unreachable_blobs for oid,size in git_blobs),fsck_returncode=unreachable_result.returncode,fsck_stderr=unreachable_result.stderr.strip())
    git_scope['binary_blob_inspection']=dict(binary_or_unknown_inspections=sum(r['format']=='binary_or_unknown' for r in git_inspections),PNG_signatures_verified=sum(r.get('binary_signature')=='PNG' for r in git_inspections),unknown_binary_inspections=[r['path'] for r in git_inspections if r['format']=='binary_or_unknown' and r.get('binary_signature')!='PNG'])
    result=dict(scope='entire CURA workspace recursively, including hidden/ignored files, code/backups/notebooks, nested archive content and every Git blob including unreachable objects',phase=label,file_count=len(inventory),workspace_file_count=sum(r['category']=='workspace' for r in inventory),suffix_counts=dict(Counter(r['suffix'] for r in inventory)),cache_candidates=cache,containers=containers,content_inspections=inspections,git_blob_count=len(git_blobs),git_candidates=git_candidates,git_content_inspections=git_inspections,git_containers=git_containers,git_scope=git_scope,source_and_backup_path_hints=hints,unresolved_ranking_candidates=unresolved,errors=errors,TREC_query_count=len(QUERIES),retrieval_executed=False)
    name='local_retrieval_audit.json' if label=='current' else 'local_retrieval_audit_'+label+'.json'
    (OUT/name).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(phase=label,files=len(inventory),inspected_formats=len(inspections),archives=len(containers),git_blobs=len(git_blobs),git_inspected_formats=len(git_inspections),git_candidates=len(git_candidates),git_scope=git_scope,unresolved_ranking_candidates=unresolved,source_and_backup_path_hint_sources=len(hints),cache_candidates=cache,errors=errors),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',default='current');args=parser.parse_args();run(args.phase)
