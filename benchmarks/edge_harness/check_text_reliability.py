# SPDX-License-Identifier: Apache-2.0
"""Isolated real-model stream saturation, cancellation and owned-worker failure checks."""

import argparse
import asyncio
import copy
import os
import sys
import time
import traceback
from pathlib import Path

from profile_local_text import save


async def run(args):
    import psutil

    from vllm_omni.edge.local.engine import LocalTextEngine
    from vllm_omni.edge.local.plan import plan_text_session

    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "start_unix": time.time(),
        "argv": sys.argv,
        "scope": (
            "LocalTextEngine consumer queue and actual owned engine-process failure; "
            "not proof of every internal queue being bounded"
        ),
    }
    engine = None
    owned = []
    save(args.out / "report.json", report)
    try:
        plan = plan_text_session(
            args.model, max_model_len=4096, max_num_seqs=4, max_num_batched_tokens=512, enforce_eager=True
        )
        report["plan"] = plan.to_dict()
        if not plan.admitted:
            report["status"] = "rejected"
            return
        oversized = plan_text_session(
            args.model, max_model_len=4096, max_num_seqs=100000, max_num_batched_tokens=512, enforce_eager=True
        )
        report["oversized_plan"] = oversized.to_dict()
        report["oversized_plan_refused"] = not oversized.admitted
        engine = LocalTextEngine(plan)
        await engine.start()
        prompt = "Describe a quiet garden with trees, flowers and a pond."

        async def request(saturate=False):
            session = engine.open_session()
            rid, stream = await engine.submit(
                session, prompt, max_tokens=128, temperature=0, ignore_eos=True, max_chunks=4
            )
            if saturate:
                # Wait for actual capacity, rather than guessing a delay from device speed.
                deadline = time.monotonic() + 30
                while stream.stats()["high_water_chunks"] < 4 and time.monotonic() < deadline:
                    await asyncio.sleep(0.05)
                report["saturated_queue_before_drain"] = stream.stats()
            events = [e.to_dict() async for e in stream]
            record = engine.records[rid].to_dict()
            record["stream"] = stream.stats()
            record["events"] = events
            engine.close_session(session.session_id)
            assert record["finished"] and not record["error"] and record["output_tokens"] == 128
            return record

        report["baseline"] = await request()
        report["saturated"] = await request(saturate=True)
        report["saturation_reached"] = report["saturated_queue_before_drain"]["high_water_chunks"] == 4
        report["same_tokens_after_saturation"] = (
            report["baseline"]["output_token_ids"] == report["saturated"]["output_token_ids"]
        )
        session = engine.open_session()
        stale = copy.deepcopy(session)
        rid, stream = await engine.submit(session, prompt, max_tokens=512, ignore_eos=True, max_chunks=4)
        deadline = time.monotonic() + 30
        while stream.stats()["high_water_chunks"] < 4 and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        report["cancel_queue_before"] = stream.stats()
        report["cancel"] = await engine.cancel(rid)
        report["cancel"]["late_events"] = [e.to_dict() async for e in stream]
        try:
            await engine.submit(stale, prompt, max_tokens=1)
        except RuntimeError as error:
            report["stale_handle_rejected"] = str(error)
        else:
            report["stale_handle_rejected"] = False
        engine.close_session(session.session_id)
        report["after_cancel"] = await request()
        save(args.out / "report.json", report)

        # Process handles come from this engine's actual manager, then must also
        # belong to this probe's process tree. Never select arbitrary system PIDs.
        client = engine._omni.engine.stage_clients[0]
        manager = client.resources.engine_manager
        processes = [p for p in manager.processes if p.is_alive()]
        descendants = {p.pid: p for p in psutil.Process().children(recursive=True)}
        assert processes and all(p.pid in descendants for p in processes)
        manager_tree = {}
        for handle in processes:
            core = descendants[handle.pid]
            manager_tree[core.pid] = core
            manager_tree.update({child.pid: child for child in core.children(recursive=True)})
        owned = [(p.pid, p.create_time()) for p in manager_tree.values()]
        target = processes[0]
        session = engine.open_session()
        rid, stream = await engine.submit(session, prompt, max_tokens=512, ignore_eos=True)
        first = await asyncio.wait_for(stream.get(), 30)
        assert first is not None and first.kind == "token", "No token before worker-failure injection"
        report["crash_first"] = first.to_dict()
        report["crash_injection"] = {
            "pid": target.pid,
            "name": target.name,
            "created": descendants[target.pid].create_time(),
            "ownership": "engine manager handle and current process descendant",
        }
        start = time.perf_counter()
        target.terminate()

        async def drain():
            return [e.to_dict() async for e in stream]

        try:
            events = await asyncio.wait_for(drain(), 30)
            report["crash_injection"].update(
                events=events,
                observed_within_30s=True,
                explicit_error_reported=bool(engine.records[rid].error) or any(e["kind"] == "error" for e in events),
                request_record=engine.records[rid].to_dict(),
            )
        except asyncio.TimeoutError:
            report["crash_injection"].update(observed_within_30s=False, error="No terminal stream event within 30s")
        report["crash_injection"]["observation_s"] = time.perf_counter() - start
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if engine is not None:
            try:
                await engine.close()
            except (Exception, asyncio.CancelledError) as error:
                report.update(status="failed", cleanup_error=repr(error))
                report.setdefault("error", f"Engine shutdown failed after probe: {error!r}")
        await asyncio.sleep(2)
        leftovers = []
        for pid, created in owned:
            try:
                p = psutil.Process(pid)
                if p.create_time() == created and p.is_running():
                    leftovers.append({"pid": pid, "created": created, "status": p.status()})
                    # Contain an isolated diagnostic's leak; report it, do not call cleanup a pass.
                    p.kill()
            except psutil.NoSuchProcess:
                pass
        report["owned_processes_requiring_harness_cleanup"] = leftovers
        report["end_unix"] = time.time()
        save(args.out / "report.json", report)
    return report["status"] == "completed"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm_omni.windows.aio import install_selector_policy

    install_selector_policy()
    raise SystemExit(0 if asyncio.run(run(args)) else 1)
