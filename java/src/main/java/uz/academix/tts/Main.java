package uz.academix.tts;

import com.fasterxml.jackson.databind.*;
import com.sun.net.httpserver.*;
import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.locks.ReentrantLock;

public final class Main {
    private static final ObjectMapper JSON=new ObjectMapper();
    private static final Set<String> FIELDS=Set.of("model","input","voice","response_format","speed","seed","temperature","cfg_weight","repetition_penalty","exaggeration","stream_format");
    private static void help(){System.out.println("""
        Uzbek TTS — offline ONNX inference (Java 17+)
        java -jar uzbek-tts.jar --text "Salom, yaxshimisiz?" --output speech.wav
        java -jar uzbek-tts.jar --text-file input.txt --output speech.wav
        java -jar uzbek-tts.jar --serve --host 127.0.0.1 --port 8000
        Options: --model DIR --cache DIR --threads N --seed N --temperature 0..2
                 --max-tokens 1..1000 --trace FILE --device cpu|auto|mps|cuda
                 --list-devices --profile DIR --help
        Model weights and ONNX Runtime are included. No Python or Docker is required.
        First use extracts verified model resources into ~/.cache/uzbek-tts.
        Default device: auto. mps uses Apple Metal via WebGPU; cuda uses NVIDIA GPU 0.
        auto prefers CUDA, then Apple Metal, then CPU and reports initialization fallback.
        Unsupported GPU operators may use CPU. CUDA requires compatible NVIDIA/CUDA/cuDNN libraries.
        This version uses the bundled default voice; audio output is WAV.
        """);}
    public static void main(String[] args)throws Exception {
        Map<String,String> flags=new HashMap<>();boolean serve=false,listDevices=false;
        Set<String> accepted=Set.of("--text","--text-file","--output","--model","--cache","--threads","--seed","--temperature","--max-tokens","--trace","--host","--port","--device","--profile");
        for(int i=0;i<args.length;i++) {
            if(args[i].equals("--help")){help();return;}
            if(args[i].equals("--serve")){serve=true;continue;}
            if(args[i].equals("--list-devices")){listDevices=true;continue;}
            if(!accepted.contains(args[i])||i+1==args.length)throw new IllegalArgumentException("Unknown or incomplete option: "+args[i]);
            flags.put(args[i],args[++i]);
        }
        if(listDevices){System.out.println(JSON.writerWithDefaultPrettyPrinter().writeValueAsString(ExecutionDevices.inventory()));return;}
        ExecutionDevices.Device.parse(flags.getOrDefault("--device","auto"));
        if(!serve && flags.containsKey("--text")==flags.containsKey("--text-file")){help();throw new IllegalArgumentException("Provide either --text or --text-file");}
        Path models=flags.containsKey("--model")?Path.of(flags.get("--model")):null;
        Path cache=flags.containsKey("--cache")?Path.of(flags.get("--cache")):null;
        long start=System.nanoTime();
        Path profile=flags.containsKey("--profile")?Path.of(flags.get("--profile")):null;
        UzbekTts tts=new UzbekTts(models,cache,Integer.parseInt(flags.getOrDefault("--threads","4")),flags.getOrDefault("--device","auto"),profile);
        System.err.println("Execution: "+JSON.writeValueAsString(tts.deviceInfo()));
        if(serve){try{serve(tts,flags);}catch(Exception failure){tts.close();throw failure;}return;}
        try(tts) {
            String text=flags.containsKey("--text")?flags.get("--text"):Files.readString(Path.of(flags.get("--text-file")),StandardCharsets.UTF_8);
            var options=new UzbekTts.Options(Long.parseLong(flags.getOrDefault("--seed","42")),Float.parseFloat(flags.getOrDefault("--temperature","0.8")),.5f,2f,.5f,Integer.parseInt(flags.getOrDefault("--max-tokens","1000")));
            var result=tts.synthesize(text,options);Path output=Path.of(flags.getOrDefault("--output","speech.wav"));
            Files.write(output,result.wav());
            if(flags.containsKey("--trace"))JSON.writerWithDefaultPrettyPrinter().writeValue(Path.of(flags.get("--trace")).toFile(),result.tokens());
            Map<String,Object> report=new LinkedHashMap<>(tts.deviceInfo());
            report.putAll(Map.of("output",output.toString(),"audio_seconds",result.samples().length/24000.0,"elapsed_seconds",(System.nanoTime()-start)/1e9,"chunks",result.tokens().size()));
            System.out.println(JSON.writeValueAsString(report));
        }
    }
    private static void serve(UzbekTts tts,Map<String,String> flags)throws IOException {
        String host=flags.getOrDefault("--host","127.0.0.1");int port=Integer.parseInt(flags.getOrDefault("--port","8000"));
        HttpServer server=HttpServer.create(new InetSocketAddress(host,port),32);
        ExecutorService executor=Executors.newFixedThreadPool(4);server.setExecutor(executor);
        ReentrantLock lock=new ReentrantLock();String apiKey=System.getenv().getOrDefault("TTS_API_KEY","");
        Set<String> origins=new HashSet<>(Arrays.asList(System.getenv().getOrDefault("TTS_CORS_ORIGINS","").split(",")));
        server.createContext("/",exchange->{
            try {
                String origin=exchange.getRequestHeaders().getFirst("Origin");
                if(origin!=null&&(origins.contains(origin)||origins.contains("*"))) {
                    exchange.getResponseHeaders().set("Access-Control-Allow-Origin",origins.contains("*")?"*":origin);
                    exchange.getResponseHeaders().set("Vary","Origin");
                    exchange.getResponseHeaders().set("Access-Control-Allow-Methods","GET,POST,OPTIONS");
                    exchange.getResponseHeaders().set("Access-Control-Allow-Headers","Authorization,Content-Type");
                }
                String path=exchange.getRequestURI().getPath(),method=exchange.getRequestMethod();
                if(method.equals("OPTIONS")){send(exchange,204,"application/json",new byte[0]);return;}
                if(method.equals("GET")&&(path.equals("/health")||path.equals("/ready"))){Map<String,Object> ready=new LinkedHashMap<>(tts.deviceInfo());ready.put("status","ready");json(exchange,200,ready);return;}
                if(!apiKey.isEmpty()&&!MessageDigest.isEqual(("Bearer "+apiKey).getBytes(StandardCharsets.UTF_8),exchange.getRequestHeaders().getFirst("Authorization")==null?new byte[0]:exchange.getRequestHeaders().getFirst("Authorization").getBytes(StandardCharsets.UTF_8))) {
                    exchange.getResponseHeaders().set("WWW-Authenticate","Bearer");error(exchange,401,"Invalid API key","unauthorized");return;
                }
                if(method.equals("GET")&&path.equals("/v1/models")) {
                    json(exchange,200,Map.of("object","list","data",List.of(Map.of("id","uzbek-chatterbox","object","model","created",0,"owned_by","local"))));return;
                }
                if(method.equals("GET")&&path.equals("/v1/audio/voices")) {json(exchange,200,Map.of("object","list","data",List.of(Map.of("id","default"))));return;}
                if(!path.equals("/v1/audio/speech")){error(exchange,404,"Not found","not_found");return;}
                if(!method.equals("POST")){error(exchange,405,"Use POST","method_not_allowed");return;}
                byte[] body=exchange.getRequestBody().readNBytes(65537);
                if(body.length>65536){error(exchange,413,"Request body is too large","invalid_request");return;}
                JsonNode request=JSON.readTree(body);
                if(request==null||!request.isObject())throw new IllegalArgumentException("JSON object required");
                request.fieldNames().forEachRemaining(name->{if(!FIELDS.contains(name))throw new IllegalArgumentException("Unsupported field: "+name);});
                if(!request.path("model").asText().equals("uzbek-chatterbox")){error(exchange,404,"Unknown model","model_not_found");return;}
                if(!request.path("input").isTextual())throw new IllegalArgumentException("input must be a string");
                for(String name:List.of("speed","seed","temperature","cfg_weight","repetition_penalty","exaggeration"))
                    if(request.has(name)&&!request.get(name).isNumber())throw new IllegalArgumentException(name+" must be numeric");
                if(request.has("seed")&&!request.get("seed").isIntegralNumber())throw new IllegalArgumentException("seed must be an integer");
                String text=request.path("input").asText("");
                if(text.isBlank()||text.codePointCount(0,text.length())>4096)throw new IllegalArgumentException("input must contain 1..4096 characters");
                if(!request.path("voice").asText("default").equals("default"))throw new IllegalArgumentException("Only voice=default is available");
                if(!request.path("response_format").asText("wav").equals("wav"))throw new IllegalArgumentException("This standalone version supports response_format=wav only");
                if(request.path("speed").asDouble(1)!=1)throw new IllegalArgumentException("This standalone version supports speed=1 only");
                if(!request.path("stream_format").asText("audio").equals("audio"))throw new IllegalArgumentException("SSE streaming is unsupported");
                var options=new UzbekTts.Options(request.path("seed").asLong(42),(float)request.path("temperature").asDouble(.8),(float)request.path("cfg_weight").asDouble(.5),(float)request.path("repetition_penalty").asDouble(2),(float)request.path("exaggeration").asDouble(.5),1000);
                if(!lock.tryLock()){exchange.getResponseHeaders().set("Retry-After","1");error(exchange,429,"Model is busy","model_busy");return;}
                try {exchange.getResponseHeaders().set("Content-Disposition","attachment; filename=\"speech.wav\"");send(exchange,200,"audio/wav",tts.synthesize(text,options).wav());}
                finally {lock.unlock();}
            } catch(IllegalArgumentException e){error(exchange,400,e.getMessage(),"invalid_request");}
              catch(com.fasterxml.jackson.core.JsonProcessingException e){error(exchange,400,"Malformed JSON","invalid_request");}
              catch(Exception e){e.printStackTrace(System.err);error(exchange,500,"Speech generation failed; see server logs","generation_failed");}
            finally {exchange.close();}
        });
        Runtime.getRuntime().addShutdownHook(new Thread(()->{server.stop(1);try{tts.close();}catch(Exception failure){failure.printStackTrace(System.err);}finally{executor.shutdownNow();}}));
        server.start();System.out.println("Uzbek TTS ready at http://"+host+":"+port);
    }
    private static void error(HttpExchange exchange,int status,String message,String code)throws IOException {
        Map<String,Object> detail=new LinkedHashMap<>();detail.put("message",message);detail.put("type",status>=500?"server_error":"invalid_request_error");detail.put("code",code);detail.put("param",null);
        json(exchange,status,Map.of("error",detail));
    }
    private static void json(HttpExchange exchange,int status,Object value)throws IOException {send(exchange,status,"application/json",JSON.writeValueAsBytes(value));}
    private static void send(HttpExchange exchange,int status,String type,byte[] data)throws IOException {
        exchange.getResponseHeaders().set("Content-Type",type);
        if(status==204){exchange.sendResponseHeaders(status,-1);return;}
        exchange.sendResponseHeaders(status,data.length);exchange.getResponseBody().write(data);
    }
}
