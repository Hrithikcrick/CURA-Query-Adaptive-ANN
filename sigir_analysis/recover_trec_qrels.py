"""Recover original NIST passage qrels; validate original query identities."""
from pathlib import Path
from collections import Counter,defaultdict
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'sigir_analysis'
BASE=ROOT/'msmarco_large/msmarco_1000000_q1000'
URL='https://trec.nist.gov/data/deep/2019qrels-pass.txt'
MIRROR='https://mirror.ir-datasets.com/2f4be390198da108f6845c822e5ada14'
EXPECTED_MD5='2f4be390198da108f6845c822e5ada14'
TARGET=ROOT/'local_data/trec2019/2019qrels-pass.txt'


def main():
    failures=[]
    if TARGET.exists():
        raw=TARGET.read_bytes();action='reused checksum-verified original-source file'
        # The initial recovery in this session used this HTTPS mirror after the
        # original NIST endpoint failed certificate validation.
        downloaded_from=MIRROR
    else:
        for candidate in [URL,MIRROR]:
            try:
                with urllib.request.urlopen(candidate,timeout=45) as response:
                    raw=response.read();assert response.status==200
                downloaded_from=candidate;break
            except Exception as exc:failures.append(dict(url=candidate,error=str(exc)))
        else:raise RuntimeError(f'Qrels download failed: {failures}')
        action='downloaded original NIST passage qrels through verified source/mirror'
    assert hashlib.md5(raw).hexdigest()==EXPECTED_MD5,'Original NIST qrels checksum mismatch'
    checksum_source=dict(resource='ir_datasets/etc/downloads.json',entry='trec-dl-2019/qrels',expected_md5=EXPECTED_MD5,expected_bytes=187092)
    spec=importlib.util.find_spec('ir_datasets')
    if spec is not None:
        registry=Path(spec.origin).parent/'etc/downloads.json'
        registry_data=json.loads(registry.read_text())
        def matching_entry(value):
            if not isinstance(value,dict):return False
            if value.get('url')==URL and value.get('expected_md5')==EXPECTED_MD5 and value.get('size_hint')==187092:return True
            return any(matching_entry(v) for v in value.values())
        assert matching_entry(registry_data),'Installed checksum registry does not confirm original qrels identity'
        checksum_source.update(package_version=importlib.metadata.version('ir_datasets'),installed_registry_match=True)
    rows=[];seen=set();qrels=defaultdict(dict)
    for line in raw.decode('utf-8-sig').splitlines():
        fields=line.split();assert len(fields)==4,fields
        qid,_,docid,grade=fields;assert (qid,docid) not in seen
        seen.add((qid,docid));grade=int(grade);assert grade in [0,1,2,3]
        rows.append((qid,docid,grade));qrels[qid][docid]=grade
    queries=[r for r in csv.DictReader((BASE/'results/queries.csv').open(encoding='utf-8-sig')) if r['source']=='trec-dl-2019']
    mapping={r['query_key']:r['raw_query_id'] for r in queries}
    assert len(mapping)==43 and len(set(mapping.values()))==43
    assert set(mapping.values())==set(qrels),'NIST query identities differ from original 43-query cohort'
    reference={r['query_key']:r for r in csv.DictReader((BASE/'research_grade_validation/exact_query_reference.csv').open(encoding='utf-8-sig'))}
    counts=[]
    for key,qid in mapping.items():
        positive=sum(v>0 for v in qrels[qid].values())
        assert positive==int(reference[key]['positive_qrel_count']),(key,positive,reference[key]['positive_qrel_count'])
        counts.append(dict(query_key=key,raw_query_id=qid,judgments=len(qrels[qid]),positive_grades_gt_0=positive,grade_0=sum(v==0 for v in qrels[qid].values()),grade_1=sum(v==1 for v in qrels[qid].values()),grade_2=sum(v==2 for v in qrels[qid].values()),grade_3=sum(v==3 for v in qrels[qid].values())))
    TARGET.parent.mkdir(parents=True,exist_ok=True)
    if not TARGET.exists():TARGET.write_bytes(raw)
    with (OUT/'trec_qrels_query_audit.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(counts[0]));writer.writeheader();writer.writerows(counts)
    info=dict(status='RECOVERED',action=action,source_url=URL,source_index='https://trec.nist.gov/data/deep2019.html',download_url=downloaded_from,download_failures=failures,transport_note='Direct NIST HTTPS had an expired-certificate error during initial recovery; verified HTTPS mirror used without disabling TLS verification.',checksum_source=checksum_source,original_md5=EXPECTED_MD5,md5_match=True,local_path=TARGET.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),judgment_rows=len(rows),query_count=len(qrels),grade_counts=dict(Counter(grade for q,d,grade in rows)),original_query_identity_match=True,saved_positive_count_match_for_all_43=True,positive_definition='grade > 0 matches original exponential-gain nDCG protocol; grade 1 remains included',FULL_ranking_metric_match='not verifiable until rankings are recovered')
    (OUT/'qrels_recovery.json').write_text(json.dumps(info,indent=2),encoding='utf-8');print(json.dumps(info,indent=2))


if __name__=='__main__':main()
