"""Real device checks; all files stay in this standalone project."""
from pathlib import Path
from collections import Counter
import argparse
import json
import subprocess
import time
import wave

ROOT=Path(__file__).resolve().parent.parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--device",required=True)
    parser.add_argument("--jar",type=Path);parser.add_argument("--label")
    parser.add_argument("--ort-version",default="1.23.2");parser.add_argument("--coreml-format",default="MLProgram");parser.add_argument("--apple-provider",default="webgpu");
    parser.add_argument("--timeout",type=int,default=300);parser.add_argument("--expect-unavailable",action="store_true")
    args=parser.parse_args();label=args.label or args.device
    if args.jar:
        command=["/usr/bin/java","-jar",str(args.jar.resolve())]
    else:
        repo=Path.home()/".m2/repository"
        jars=[repo/f"com/microsoft/onnxruntime/onnxruntime/{args.ort_version}/onnxruntime-{args.ort_version}.jar"]
        jars += [repo/f"com/fasterxml/jackson/core/{name}/2.18.2/{name}-2.18.2.jar" for name in ("jackson-databind","jackson-core","jackson-annotations")]
        command=["/usr/bin/java","-Duzbek.tts.coreml.format="+args.coreml_format,"-Duzbek.tts.apple.provider="+args.apple_provider,"-cp",":".join([str(ROOT/"java/target/classes"),*[str(p) for p in jars]]),"uz.academix.tts.Main"]
    log=ROOT/f"docs/device-{label}.log";output=ROOT/f"samples/device-{label}.wav"
    profile=ROOT/f"docs/device-profile-{label}"
    command += ["--device",args.device,"--model",str(ROOT/"models"),"--cache",str(ROOT/"build/device-cache"),
        "--profile",str(profile),"--text","Ha.","--temperature","0","--output",str(output)]
    started=time.time()
    try:
        with log.open("w") as stream:
            result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=args.timeout)
        code=result.returncode
    except subprocess.TimeoutExpired:code=124
    text=log.read_text();report={"returncode":code,"seconds":round(time.time()-started,2),"requested_device":args.device,"ort_version":args.ort_version,"coreml_format":args.coreml_format,"apple_provider":args.apple_provider}
    for line in text.splitlines():
        if line.startswith("{"):
            try:report["result"]=json.loads(line)
            except json.JSONDecodeError:pass
    providers={}
    for graph in ("embeddings","language_model","decoder"):
        paths=sorted(profile.glob(graph+"_*.json"),key=lambda p:p.stat().st_mtime)
        if paths:
            try:events=json.loads(paths[-1].read_text())
            except json.JSONDecodeError:events=[]
            providers[graph]=dict(Counter(event.get("args",{}).get("provider") for event in events if event.get("args",{}).get("provider")))
    report["profile_provider_events"]=providers
    if args.expect_unavailable:
        assert code!=0 and ("requires" in text or "Cannot initialize" in text),text[-5000:]
        report["status"]="unavailable_as_expected"
    elif code==0:
        with wave.open(str(output)) as audio:
            assert audio.getframerate()==24000 and audio.getnframes()>0
            report["audio_seconds"]=audio.getnframes()/audio.getframerate()
        report["status"]="passed"
    else:report["status"]="failed"
    (ROOT/f"docs/device-{label}-result.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)
    print(text[-5000:],flush=True)
    if report["status"]=="failed":raise SystemExit(1)

if __name__=="__main__":main()
