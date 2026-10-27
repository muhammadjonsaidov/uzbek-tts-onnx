"""Exercise the standalone JAR's HTTP API with standard-library clients."""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import json
import os
import time
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import wave

ROOT=Path(__file__).resolve().parent.parent
URL="http://127.0.0.1:18001"
KEY="standalone-smoke-key"

def request(path,body=None,auth=True,method=None,headers=None):
    h={"Content-Type":"application/json"}
    if auth:h["Authorization"]="Bearer "+KEY
    h.update(headers or {})
    req=Request(URL+path,data=None if body is None else json.dumps(body).encode(),headers=h,method=method)
    try:
        with urlopen(req,timeout=240) as response:return response.status,dict(response.headers),response.read()
    except HTTPError as error:return error.code,dict(error.headers),error.read()

def main():
    checks=[]
    assert request("/health",auth=False)[0]==200;checks.append("health")
    code,_,ready_body=request("/ready",auth=False)
    ready=json.loads(ready_body)
    assert code==200
    expected=os.environ.get("TTS_TEST_EXPECT_DEVICE")
    if expected:assert ready["device"]==expected,ready
    checks.append("ready")
    assert request("/v1/models",auth=False)[0]==401;checks.append("bearer_auth")
    code,_,data=request("/v1/models");assert code==200 and json.loads(data)["data"][0]["id"]=="uzbek-chatterbox";checks.append("models")
    assert json.loads(request("/v1/audio/voices")[2])["data"]==[{"id":"default"}];checks.append("voices")
    code,headers,_=request("/v1/audio/speech",auth=False,method="OPTIONS",headers={"Origin":"http://localhost:3000","Access-Control-Request-Method":"POST"})
    assert code==204 and headers["Access-control-allow-origin"]=="http://localhost:3000";checks.append("cors")
    base={"model":"uzbek-chatterbox","input":"Ha.","voice":"default","response_format":"wav","temperature":0}
    for field,value in [("input"," "),("input",5),("input","a"*4097),("voice","other"),("response_format","mp3"),("speed",2),("seed","bad"),("seed",-1),("temperature",-1),("instructions","ignored?")]:
        assert request("/v1/audio/speech",{**base,field:value})[0]==400,field
        checks.append("reject_"+field)
    assert request("/v1/audio/speech",{**base,"model":"other"})[0]==404;checks.append("unknown_model")
    started=time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(request,"/v1/audio/speech",base) for _ in range(2)]
        results=[future.result() for future in futures]
    assert sorted(result[0] for result in results)==[200,429]
    successful=next(result for result in results if result[0]==200)
    assert successful[1]["Content-type"]=="audio/wav"
    with wave.open(BytesIO(successful[2])) as audio:
        assert audio.getframerate()==24000 and audio.getnchannels()==1 and audio.getnframes()>0
        seconds=audio.getnframes()/audio.getframerate()
    busy=next(result for result in results if result[0]==429)
    assert busy[1]["Retry-after"]=="1"
    (ROOT/"samples/standalone-http.wav").write_bytes(successful[2])
    checks += ["real_wav","concurrent_busy_429"]
    report={"status":"passed","checks":checks,"device":ready.get("device"),"provider":ready.get("provider"),"audio_seconds":seconds,"inference_seconds":round(time.time()-started,2)}
    (ROOT/"docs/validation-http.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
