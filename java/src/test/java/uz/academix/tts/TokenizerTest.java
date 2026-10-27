package uz.academix.tts;
import com.fasterxml.jackson.databind.*;
import java.nio.file.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
class TokenizerTest {
    @Test void matchesOriginalRustTokenizer()throws Exception {
        Path root=Path.of("..").toAbsolutePath();
        var tokenizer=new TextTokenizer(root.resolve("models/tokenizer.json"));
        for(JsonNode test:new ObjectMapper().readTree(root.resolve("tests/tokenizer-cases.json").toFile())) {
            String text=test.path("text").asText();
            long[] expected=new long[test.path("ids").size()];
            for(int i=0;i<expected.length;i++)expected[i]=test.path("ids").get(i).asLong();
            assertEquals(test.path("normalized").asText(),TextTokenizer.normalize(text),text);
            assertArrayEquals(expected,tokenizer.encode(text),text);
        }
    }
    @Test void splitsLongInputWithoutLosingCharacters() {
        var chunks=TextTokenizer.split("a".repeat(500));
        assertTrue(chunks.stream().allMatch(s->s.length()<=140));
        assertEquals("a".repeat(500),String.join("",chunks));
        assertThrows(IllegalArgumentException.class,()->TextTokenizer.split("  "));
    }
    @Test void rejectsInvalidOptions() {
        assertThrows(IllegalArgumentException.class,()->new UzbekTts.Options(42,Float.NaN,.5f,2,.5f,1000));
        assertThrows(IllegalArgumentException.class,()->new UzbekTts.Options(42,.8f,.5f,1,.5f,1000));
    }
}
