"""Does the NPU-steps + GPU-vocoder split hold across the Snapdragon family? Same two graphs on a laptop-class part,
an automotive board and a low-end IoT board, next to the two phones."""
import json, qai_hub as hub
S="/tmp/claude-2005/-data-zhoutaichang-embedding-infer/e695f57b-0163-4287-b1be-1d0ec9330768/scratchpad/qnn"
meta=json.load(open(f"{S}/talker_step2_L256.json")); NL,NKV,D,H,L=meta["NL"],meta["NKV"],meta["D"],meta["H"],meta["L"]
tspec={"x":((1,1,H),"float32"),"cos":((1,1,1,D),"float32"),"sin":((1,1,1,D),"float32"),"mask":((1,1,1,L+1),"float32")}
for i in range(NL): tspec[f"k_cache_{i}"]=((1,NKV,L,D),"float32"); tspec[f"v_cache_{i}"]=((1,NKV,L,D),"float32")
graphs={"talker_step2": (list(json.load(open(f"{S}/aihub/jobs7.json")).values())[0]["model_id"], tspec, "qnn_context_binary", "npu"),
        "code2wav":     (json.load(open(f"{S}/aihub/jobs.json"))["code2wav_htp_c25@Samsung Galaxy S24"]["model_id"], {"quantized":((1,512,97),"float32")}, "tflite", "gpu")}
devices=["Snapdragon X Elite CRD","SA8775P ADP","Dragonwing RB3 Gen 2 Vision Kit"]
res={}; pend={}
for dname in devices:
    for gname,(mid,spec,rt,unit) in graphs.items():
        try:
            dev=hub.Device(dname)
            cj=hub.submit_compile_job(model=hub.get_model(mid), device=dev, input_specs=spec, name=f"{gname} {rt}@{dname}", options=f"--target_runtime {rt}"+(" --quantize_full_type float16" if rt=="qnn_context_binary" else ""))
            pend[(dname,gname,rt,unit)]=cj; print("compile submitted", dname, gname, rt, cj.job_id, flush=True)
        except Exception as e: print("rejected", dname, gname, str(e)[:200], flush=True)
prof={}
for key,cj in pend.items():
    dname,gname,rt,unit=key; st=cj.wait(); print("compile", dname, gname, st.code, (st.message or "")[:120], flush=True)
    if not st.success: continue
    try:
        pj=hub.submit_profile_job(model=cj.get_target_model(), device=hub.Device(dname), name=f"{gname} on {dname} {unit}", options=f"--compute_unit {unit}")
        prof[key]=pj
    except Exception as e: print("profile rejected", dname, gname, str(e)[:160], flush=True)
for key,pj in prof.items():
    dname,gname,rt,unit=key; st=pj.wait()
    if st.success:
        p=pj.download_profile(); ms=p["execution_summary"]["estimated_inference_time"]/1e3
        res[f"{gname}|{dname}|{rt}|{unit}"]=ms
        json.dump(p, open(f"{S}/aihub/profile_{gname}_{dname.replace(' ','_')}_{rt}_{unit}.json","w"))
        print(f"  {gname} on {dname} ({rt}/{unit}): {ms:.1f} ms" + (f" -> {ms/25:.1f} ms per frame" if gname=="code2wav" else ""), flush=True)
    else: print(f"  {gname} on {dname} failed:", (st.message or "")[:160], flush=True)
    json.dump(res, open(f"{S}/aihub/device_matrix.json","w"), indent=1)
print("done", flush=True)
