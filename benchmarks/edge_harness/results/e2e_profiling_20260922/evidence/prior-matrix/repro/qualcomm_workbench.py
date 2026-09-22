"""Explicit AI Hub test facility; never imported by the local Omni runtime.

Pass an API token on stdin or use an already configured SDK profile. Secrets
are never written to results. Inventory is read-only; profile/infer commands
submit device jobs only for an explicitly supplied existing model ID.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import qai_hub as hub


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=("inventory", "profile", "infer", "collect"))
    parser.add_argument("--token-stdin", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--device", default="Samsung Galaxy S25")
    parser.add_argument("--options", default="--compute_unit npu")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--reference-job", help="Reuse a prior inference job's exact dataset")
    parser.add_argument("--job")
    args = parser.parse_args()
    client = hub.Client(hub.ClientConfig(api_token=sys.stdin.readline().strip())) if args.token_stdin else hub.Client()
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "qai_hub_version": importlib.metadata.version("qai-hub"),
        "facility": "Qualcomm AI Hub Workbench",
        "command": args.command,
    }
    if args.command == "inventory":
        devices = client.get_devices()
        record["devices"] = [{"name": d.name, "os": d.os, "attributes": d.attributes} for d in devices]
        jobs = client.get_job_summaries(limit=100)
        record["recent_jobs"] = [
            {
                "job_id": j.job_id,
                "name": j.name,
                "date": str(j.date),
                "status": j.status.code,
                "type": str(j.job_type),
                "device": j.device_name,
                "url": j.url,
            }
            for j in jobs
        ]
        models = client.get_models(limit=100)
        record["models"] = [{"model_id": m.model_id, "name": m.name} for m in models]
        print(f"Connected: {len(devices)} device entries, {len(jobs)} recent jobs, {len(models)} models")
    elif args.command in ("profile", "infer"):
        if not args.model:
            parser.error("--model is required")
        kwargs = dict(
            model=client.get_model(args.model),
            device=hub.Device(args.device),
            name=f"omni-boundary-validation-{args.command}",
            options=args.options,
        )
        if args.command == "infer":
            import numpy as np

            if bool(args.inputs) == bool(args.reference_job):
                parser.error("provide exactly one of --inputs or --reference-job")
            if args.reference_job:
                reference = client.get_job(args.reference_job)
                if not isinstance(reference, hub.InferenceJob):
                    parser.error("reference must be an inference job")
                kwargs["inputs"] = reference.inputs
                record["reference_job"] = reference.job_id
                record["input_dataset_id"] = reference.inputs.dataset_id
            else:
                with np.load(args.inputs, allow_pickle=False) as data:
                    kwargs["inputs"] = {name: [data[name]] for name in data.files}
            job = client.submit_inference_job(**kwargs)
        else:
            job = client.submit_profile_job(**kwargs)
        record.update(
            job_id=job.job_id, url=job.url, model_id=args.model, requested_device=args.device, options=args.options
        )
        print(f"Submitted {args.command}: {job.job_id}")
    else:
        if not args.job:
            parser.error("--job is required")
        job = client.get_job(args.job)
        status = job.get_status()
        record.update(
            job_id=job.job_id,
            url=job.url,
            status=status.code,
            status_message=status.message,
            options=job.options,
            hub_version=job.hub_version,
        )
        if hasattr(job, "device"):
            record["device"] = {"name": job.device.name, "os": job.device.os, "attributes": job.device.attributes}
        if hasattr(job, "model"):
            record["model_id"] = job.model.model_id
        if isinstance(job, hub.ProfileJob):
            record["shapes"] = job.shapes
        if isinstance(job, hub.InferenceJob):
            record["input_dataset_id"] = job.inputs.dataset_id
        if status.success:
            if isinstance(job, hub.ProfileJob):
                record["profile"] = job.download_profile()
            elif isinstance(job, hub.InferenceJob):
                import numpy as np

                outputs = job.download_output_data()
                output_path = args.out.with_suffix(".npz")
                np.savez(
                    output_path,
                    **{f"{key}__{i}": value for key, values in outputs.items() for i, value in enumerate(values)},
                )
                record["outputs"] = output_path.name
            elif isinstance(job, hub.CompileJob):
                target = job.get_target_model()
                record["target_model_id"] = target.model_id if target else None
        print(f"Job {job.job_id}: {status.code}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
