package uz.academix.tts;

import org.junit.jupiter.api.Test;
import java.util.concurrent.atomic.AtomicInteger;
import static org.junit.jupiter.api.Assertions.*;

class RuntimeResourcesTest {
    @Test void closesRegisteredResourcesBeforeReturning(){
        var registry=new RuntimeResources.Registry();var count=new AtomicInteger();
        AutoCloseable resource=()->{count.incrementAndGet();registry.unregister(null);};
        registry.register(resource);registry.register(resource);registry.closeAll();
        assertEquals(1,count.get());
        assertThrows(IllegalStateException.class,()->registry.register(resource));
    }
    @Test void skipsExplicitlyClosedResources(){
        var registry=new RuntimeResources.Registry();var count=new AtomicInteger();
        AutoCloseable resource=count::incrementAndGet;
        registry.register(resource);registry.unregister(resource);registry.closeAll();
        assertEquals(0,count.get());
    }
    @Test void oneCleanupFailureDoesNotSkipOtherResources(){
        var registry=new RuntimeResources.Registry();var count=new AtomicInteger();
        registry.register(()->{throw new Exception("expected cleanup failure");});
        registry.register(count::incrementAndGet);registry.closeAll();assertEquals(1,count.get());
    }
}
