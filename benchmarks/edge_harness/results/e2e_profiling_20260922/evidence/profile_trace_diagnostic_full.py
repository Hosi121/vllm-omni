# SPDX-License-Identifier: Apache-2.0
"""Separate paired untraced/traced requests through Omni's existing profiler RPC."""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback

from profile_local_text import save


async def run(args):
    from vllm_omni import AsyncOmni
    from vllm_omni.edge.local.engine import apply_runtime_env
    from vllm_omni.edge.local.manifest import runtime_versions

    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    trace_dir = (args.out / 'traces').resolve()
    report = {'status': 'running', 'start_unix': time.time(), 'argv': sys.argv,
              'python': sys.executable, 'platform': platform.platform(),
              'runtime': runtime_versions().to_dict(), 'environment': apply_runtime_env(),
              'scope': 'diagnostic paired profiling overhead; not steady-state latency percentiles',
              'requests': []}
    config = {'profiler': 'torch', 'torch_profiler_dir': str(trace_dir),
              'torch_profiler_with_stack': False, 'torch_profiler_record_shapes': True,
              'torch_profiler_with_memory': True, 'torch_profiler_use_gzip': True,
              'torch_profiler_dump_cuda_time_total': False}
    report['profiler_config'] = config
    save(args.out/'report.json', report)
    omni = None
    profiling = False
    try:
        params = None
        if args.kind == 'spark':
            from vllm import SamplingParams
            from vllm_omni.edge.local.plan import plan_text_session
            plan = plan_text_session(args.model, max_model_len=4096, max_num_seqs=4,
                                     max_num_batched_tokens=512, enforce_eager=True)
            report['plan'] = plan.to_dict()
            if not plan.admitted:
                report['status'] = 'rejected'
                return
            kwargs = plan.engine_kwargs
            prompt = 'The garden has trees, flowers, and a small pond. Describe the garden in detail.'
            params = SamplingParams(temperature=0, max_tokens=128, ignore_eos=True)
        else:
            from transformers import AutoTokenizer
            from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig
            from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import Qwen3TTSPromptEmbedsBuilder
            cfg = Qwen3TTSConfig.from_pretrained(args.model)
            tok = AutoTokenizer.from_pretrained(args.model)
            info = {'task_type': ['CustomVoice'], 'text': ['Hello, this is a speech test.'],
                    'language': ['English'], 'speaker': ['Vivian']}
            length = Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
                additional_information=info, task_type='CustomVoice',
                tokenize_prompt=lambda x: tok(x, padding=False)['input_ids'],
                codec_language_id=cfg.talker_config.codec_language_id,
                spk_is_dialect=cfg.talker_config.spk_is_dialect)
            prompt = {'prompt_token_ids': [0]*length, 'additional_information': info}
            kwargs = {'deploy_profile': 'edge', 'stage_init_timeout': 180}
        start = time.perf_counter()
        omni = AsyncOmni(model=args.model, **kwargs, profiler_config=config)
        report['startup_s'] = time.perf_counter()-start

        async def request(phase, number):
            start = time.perf_counter()
            outputs = 0
            finished = False
            audio_samples = 0
            async for output in omni.generate(prompt, sampling_params=params,
                                               request_id=f'{phase}-{number}'):
                outputs += 1
                finished = finished or output.finished
                if args.kind == 'tts' and output.outputs:
                    mm = output.outputs[0].multimodal_output
                    audio = mm.get('audio') if mm else None
                    parts = audio if isinstance(audio, list) else [audio]
                    audio_samples += sum(int(part.numel()) for part in parts if part is not None)
            row = {'phase': phase, 'index': number, 'wall_s': time.perf_counter()-start,
                   'events': outputs, 'finished': finished, 'audio_samples': audio_samples}
            report['requests'].append(row)
            save(args.out/'report.json', report)
            if not finished or not outputs:
                raise RuntimeError(f'Incomplete diagnostic request: {row}')

        await request('warmup', 0)
        for i in range(3):
            await request('untraced_before', i)
        report['start_rpc'] = await omni.start_profile(profile_prefix='e2e-diagnostic')
        profiling = True
        for i in range(3):
            await request('traced', i)
        report['stop_rpc'] = await omni.stop_profile()
        profiling = False
        for i in range(3):
            await request('untraced_after', i)
        medians = {phase: statistics.median(r['wall_s'] for r in report['requests'] if r['phase'] == phase)
                   for phase in ('untraced_before', 'traced', 'untraced_after')}
        report['median_wall_s'] = medians
        report['traced_over_untraced_before'] = medians['traced']/medians['untraced_before']
        report['traced_over_untraced_after'] = medians['traced']/medians['untraced_after']
        if args.kind == 'tts':
            report['overhead_caveat'] = 'Same prompt but speech generation can vary; compare audio_samples alongside wall-time ratios.'
        report['traces'] = [{'path': str(f.relative_to(args.out)), 'bytes': f.stat().st_size,
                             'sha256': hashlib.sha256(f.read_bytes()).hexdigest()}
                            for f in trace_dir.rglob('*') if f.is_file()]
        if not report['traces']:
            raise RuntimeError('Profiler RPC returned but no trace artifact was written')
        report['status'] = 'completed'
    except Exception as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if profiling and omni is not None:
            try:
                await omni.stop_profile()
            except Exception as error:
                report['stop_error'] = repr(error)
        if omni is not None:
            omni.shutdown()
        report['end_unix'] = time.time()
        save(args.out/'report.json', report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=['spark', 'tts'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    from vllm_omni.windows.aio import install_selector_policy
    install_selector_policy()
    asyncio.run(run(args))
