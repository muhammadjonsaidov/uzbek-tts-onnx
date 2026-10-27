package uz.academix.tts;

import java.util.*;

/** Orders model cleanup before ONNX Runtime releases its shared GPU context. */
public final class RuntimeResources {
    private static final Registry RESOURCES=new Registry();
    private RuntimeResources(){}
    static void register(AutoCloseable resource){RESOURCES.register(resource);}
    static void unregister(AutoCloseable resource){RESOURCES.unregister(resource);}
    /** Called by the bundled ONNX Runtime environment shutdown hook. */
    public static void closeAll(){RESOURCES.closeAll();}

    static final class Registry {
        private final Set<AutoCloseable> resources=Collections.newSetFromMap(new IdentityHashMap<>());
        private boolean closing;
        synchronized void register(AutoCloseable resource){
            if(closing)throw new IllegalStateException("ONNX Runtime is shutting down");
            resources.add(resource);
        }
        synchronized void unregister(AutoCloseable resource){resources.remove(resource);}
        void closeAll(){
            List<AutoCloseable> snapshot;
            synchronized(this){closing=true;snapshot=new ArrayList<>(resources);}
            for(AutoCloseable resource:snapshot)try{resource.close();}
            catch(Exception failure){System.err.println("Error closing TTS resource: "+failure);}
        }
    }
}
