"""External trace endpoint with explicit requirements; generated software fixture only."""
import argparse
import json
from pathlib import Path
import numpy as np

from audit_denoiser.adapters import validate_contract_method_output
from audit_denoiser.contracts import DatasetContract, MethodAdapterContract, MetricContract, evaluate_metric, write_json
from audit_denoiser.schema import sha256


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    array=np.arange(128,dtype=np.float32).reshape(64,2)
    array_path=output/'traces.npy'
    np.save(array_path,array)
    dataset=DatasetContract('external-trace-fixture','waveform','TR',array.shape,'float32','arbitrary',100.0,
        'fixture','record',('record','channel','sample'),'none','reference_free','none','not_applicable',
        {'source':'generated external integration example'}).validate()
    method=MethodAdapterContract('external-producer','example-source-v1',None,'none','trace_only',
        ('external_integrity',),(),('traces',)).validate()
    validation=validate_contract_method_output(method,array_path,dataset)
    policy=MetricContract('external_finite_fraction','external_integrity',('reference_free',),('trace_only',))
    calls=[]
    def numerical():
        calls.append('external_finite_fraction')
        return float(np.isfinite(array).mean())
    def forbidden():
        raise AssertionError('incompatible metric callback must not execute')
    common=dict(run_id='external-fixture-run',metric_version='example-v1',units='fraction',taxonomy_version='example-v1',
        hashes={'output_sha256':sha256(array_path)})
    rows=[evaluate_metric(policy.metric_id,dataset,method,analysis=numerical,metric_contract=policy,**common),
          evaluate_metric('unregistered_endpoint',dataset,method,analysis=forbidden,**common),
          evaluate_metric('waveform_pearson_to_clean',dataset,method,analysis=forbidden,**common)]
    payload={'fixture_only':True,'validation':validation,'analysis_callbacks':calls,'results':[r.to_dict() for r in rows]}
    write_json(payload,output/'results.json')
    return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.out),indent=2))
