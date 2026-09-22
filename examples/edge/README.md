# Local graph example

See [the backend contract and deployment guide](../../docs/design/omni_edge_backends.md).

From a configured Omni environment:

```bash
python examples/edge/run_graph_stage.py /path/to/manifest.json \
  --python /path/to/ort-venv/bin/python --report /path/to/run.json
```

The worker environment needs `numpy`, `onnx` and `onnxruntime`. For Windows,
use `--windows-worker` and the controller-visible path to `python.exe`.
`--ep vitisai` or `--ep dml` selects a vendor provider; declare every permitted
provider using `--allowed-provider`. No provider substitution is hidden.

The example reads fixed-bucket inputs from the bundle. It reports backend
execution only; use your own model adapter and quality fixtures for a model claim.
