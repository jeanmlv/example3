#!/usr/bin/env python3
"""Reconcile QUASAR Thea.View rawdata and processed video inventory.
Offline only. Requires pandas. Run:
python 03_validate_video_coverage.py --input video_inventory_output/all_video_inventory.csv

Creates an auditable, conservative report. Does not claim content-level uniqueness.
"""
import argparse
import re
import json
from pathlib import Path
import pandas as pd

VISITS = ['Screening','I-12','I-24','M-44','M-92','M-140','M-188','M-236']

def norm_name(s):
    s = re.sub(r'\.[^.]+$', '', str(s).strip(), flags=re.I)
    s = re.sub(r'(?i)([._ -]+transcoded)+$', '', s)
    return re.sub(r'\s+', ' ', s).casefold().strip()

def norm_date(s):
    v = pd.to_datetime(str(s).strip(), errors='coerce', dayfirst=True)
    return v.strftime('%Y-%m-%d') if pd.notna(v) else str(s).strip().casefold()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='video_inventory_output/all_video_inventory.csv')
    ap.add_argument('--output-dir',default='video_validation_output')
    a=ap.parse_args()
    path=Path(a.input)
    if not path.is_file(): ap.error(f'Input missing: {path.resolve()}')
    d=pd.read_csv(path,dtype=str,keep_default_na=False,encoding='utf-8-sig')
    needed=['study','data_type','video_class','site','patient_id','visit','examination_date','anatomical_region','filename','item_id']
    absent=[x for x in needed if x not in d]
    if absent: ap.error('Missing columns: '+', '.join(absent))
    if d.empty: ap.error('Empty input')
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    for c in needed+['processor','batch','relative_path']:
        if c not in d:d[c]=''
        d[c]=d[c].astype(str).str.strip()
    d['name_stem']=d.filename.map(norm_name)
    d['date_key']=d.examination_date.map(norm_date)
    d['patient_key']=d.site+'::'+d.patient_id
    d['valid_patient']=(d.site!='')&(d.patient_id!='')
    d['recording_key']=d[['study','site','patient_id','visit','date_key','anatomical_region','name_stem']].agg('|'.join,axis=1)
    d['visit_patient_key']=d[['study','site','patient_id','visit']].agg('|'.join,axis=1)
    d['raw_priority']=d.video_class.map({'RAW_ORIGINAL':0,'RAW_UNCLASSIFIED':1,'RAW_TRANSCODED':2}).fillna(3)
    d=d.sort_values(['data_type','recording_key','raw_priority','item_id'],kind='stable')
    d['technical_representative']=~d.duplicated(['data_type','recording_key'])
    representatives=d[d.technical_representative].copy()
    raw_keys=set(representatives.loc[representatives.data_type=='rawdata','recording_key'])
    proc_keys=set(representatives.loc[representatives.data_type=='processed','recording_key'])
    representatives['cross_source_status']=representatives.apply(lambda r: 'MATCHED_BOTH' if r.recording_key in raw_keys & proc_keys else ('RAW_ONLY' if r.data_type=='rawdata' else 'PROCESSED_ONLY'),axis=1)
    representatives['review_reason']=''
    representatives.loc[~representatives.valid_patient,'review_reason']='MISSING_PATIENT_OR_SITE'
    representatives.loc[representatives.video_class.eq('RAW_UNCLASSIFIED'),'review_reason'] += ';RAW_UNCLASSIFIED'
    representatives.loc[representatives.video_class.eq('RAW_TRANSCODED'),'review_reason'] += ';TRANSCODED_WITHOUT_MATCHED_ORIGINAL'
    representatives.loc[~representatives.visit.isin(VISITS),'review_reason'] += ';NON_SCHEDULED_VISIT'
    representatives.loc[representatives.cross_source_status.ne('MATCHED_BOTH'),'review_reason'] += ';NO_EXACT_CROSS_SOURCE_MATCH'
    # Patient+visit is only a weak match, never evidence of identical video.
    visit_raw=set(d.loc[d.data_type=='rawdata','visit_patient_key'])
    visit_proc=set(d.loc[d.data_type=='processed','visit_patient_key'])
    representatives['patient_visit_in_other_source']=representatives.apply(lambda r: r.visit_patient_key in (visit_proc if r.data_type=='rawdata' else visit_raw),axis=1)
    representatives.to_csv(out/'02_reconciled_recordings.csv',index=False,encoding='utf-8-sig')
    representatives[representatives.review_reason!=''].to_csv(out/'03_records_for_review.csv',index=False,encoding='utf-8-sig')

    rows=[]
    for study in sorted(d.study.unique()):
        for v in VISITS + sorted(set(d.loc[d.study==study,'visit'])-set(VISITS)):
            row={'study':study,'visit':v,'visit_type':'SCHEDULED' if v in VISITS else 'UNSCHEDULED'}
            for typ in ['rawdata','processed']:
                subset=d[(d.study==study)&(d.visit==v)&(d.data_type==typ)]
                rec=representatives[(representatives.study==study)&(representatives.visit==v)&(representatives.data_type==typ)]
                row[f'{typ}_file_items']=len(subset)
                row[f'{typ}_technical_recordings_est']=len(rec)
                row[f'{typ}_unique_patients_at_visit']=rec.loc[rec.valid_patient,'patient_key'].nunique()
                row[f'{typ}_exact_cross_source_matched_recordings']=int((rec.cross_source_status=='MATCHED_BOTH').sum())
                row[f'{typ}_no_exact_match_recordings']=int((rec.cross_source_status!='MATCHED_BOTH').sum())
                if v in VISITS:
                    prior=VISITS[:VISITS.index(v)+1]
                    cumulative=representatives[(representatives.study==study)&(representatives.data_type==typ)&(representatives.visit.isin(prior))&representatives.valid_patient]
                    row[f'{typ}_cumulative_unique_patients']=cumulative.patient_key.nunique()
                else: row[f'{typ}_cumulative_unique_patients']=''
            rows.append(row)
    table=pd.DataFrame(rows)
    table.to_csv(out/'01_pablo_summary_by_visit.csv',index=False,encoding='utf-8-sig')
    checks=[]
    for study in sorted(d.study.unique()):
        for typ in ['rawdata','processed']:
            sub=d[(d.study==study)&(d.data_type==typ)]
            rec=representatives[(representatives.study==study)&(representatives.data_type==typ)]
            checks.append({'study':study,'data_type':typ,'file_items':len(sub),
                           'technical_recordings_est':len(rec),
                           'unique_patients_all_visits':rec.loc[rec.valid_patient,'patient_key'].nunique(),
                           'duplicate_file_items_by_technical_key':len(sub)-len(rec),
                           'exact_cross_source_matched_recordings':int((rec.cross_source_status=='MATCHED_BOTH').sum()),
                           'no_exact_cross_source_match':int((rec.cross_source_status!='MATCHED_BOTH').sum()),
                           'raw_unclassified_file_items':int((sub.video_class=='RAW_UNCLASSIFIED').sum()),
                           'distinct_processors':sub.processor[sub.processor!=''].nunique(),
                           'distinct_patient_ids':sub.patient_id[sub.patient_id!=''].nunique(),
                           'distinct_site_patient_pairs':sub.loc[(sub.site!='')&(sub.patient_id!=''),'patient_key'].nunique()})
    qa=pd.DataFrame(checks)
    qa.to_csv(out/'04_quality_checks.csv',index=False,encoding='utf-8-sig')
    methodology='''# QUASAR video reconciliation — interpretation

- **File items**: API inventory rows. Includes original, transcoded, and processed files.
- **Technical recordings (estimated)**: unique study + site + patient + visit + normalized examination date + anatomical region + filename stem, separately for rawdata/processed. Raw original takes precedence over raw transcoded. Matching is based on metadata only, not video content.
- **Exact cross-source match**: same technical key in both rawdata and processed. A missing exact match is NOT proof of a new clinical recording (names/dates may differ).
- **Unique patients at visit**: unique site + patient_id with >=1 technical recording in that visit.
- **Cumulative unique patients**: unique site + patient_id with >=1 recording in this or an earlier *scheduled* visit. NOT patients with a video at every preceding visit.
- **Scheduled order**: Screening, I-12, I-24, M-44, M-92, M-140, M-188, M-236. Provisional; verify against QUASAR protocol.
- Unscheduled visits are counted individually but excluded from cumulative results.
- No expected-population denominator is available; no completeness percentages.
- Confirm the validity of site+patient_id, filenames, and the 252 RAW_UNCLASSIFIED items before claiming clinical video uniqueness.
- Confirm extraction_summary.json reports COMPLETE for both roots and zero API errors.
- This analysis is offline and does not download videos or contact Thea.View.
'''
    (out/'05_methodology_and_limitations.md').write_text(methodology,encoding='utf-8')
    summary={'input_rows':len(d),'studies':sorted(d.study.unique()),'exact_matched_recording_keys':len(raw_keys & proc_keys),'raw_only_technical_keys':len(raw_keys-proc_keys),'processed_only_technical_keys':len(proc_keys-raw_keys),'warnings':['Metadata-based matching; not video content validation','Visit order provisional','Unscheduled visits excluded from cumulative patient count']}
    (out/'06_validation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('DONE —',out.resolve())
    print(qa.to_string(index=False))
    print('Exact matching technical keys:',summary['exact_matched_recording_keys'])
    print('Raw-only keys:',summary['raw_only_technical_keys'],'Processed-only keys:',summary['processed_only_technical_keys'])
    print('Send 01_pablo_summary_by_visit.csv and 04_quality_checks.csv for review.')
if __name__=='__main__':main()
