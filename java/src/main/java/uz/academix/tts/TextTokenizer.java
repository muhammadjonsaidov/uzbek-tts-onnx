package uz.academix.tts;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import java.util.regex.*;

final class TextTokenizer {
    private final Map<String,Integer> vocabulary = new HashMap<>();
    private final Map<String,Integer> ranks = new HashMap<>();
    private final Map<String,Integer> specials = new HashMap<>();
    private final Pattern specialPattern;
    private static final Pattern WORDS = Pattern.compile("(?U)\\w+|[^\\w\\s]+");

    TextTokenizer(Path file) throws IOException {
        JsonNode json = new ObjectMapper().readTree(file.toFile());
        json.path("model").path("vocab").fields().forEachRemaining(e -> vocabulary.put(e.getKey(),e.getValue().asInt()));
        int rank=0;
        for(JsonNode merge:json.path("model").path("merges")) {
            String[] pair=merge.isArray()?new String[]{merge.get(0).asText(),merge.get(1).asText()}:merge.asText().split(" ",2);
            ranks.put(pair[0]+"\0"+pair[1],rank++);
        }
        for(JsonNode entry:json.path("added_tokens")) specials.put(entry.path("content").asText(),entry.path("id").asInt());
        specialPattern=Pattern.compile(specials.keySet().stream().sorted(Comparator.comparingInt(String::length).reversed())
            .map(Pattern::quote).reduce((a,b)->a+"|"+b).orElse("(?!)"));
    }

    static String normalize(String value) {
        String text=value.replaceAll("(?U)\\s+"," ").strip();
        if(text.isEmpty())throw new IllegalArgumentException("Text is empty");
        int first=text.codePointAt(0),count=Character.charCount(first);
        if(Character.isLowerCase(first)) text=text.substring(0,count).toUpperCase(Locale.ROOT)+text.substring(count);
        String[][] replacements={{"...",", "},{"…",", "},{":",","},{" - ",", "},{";",","},{"—","-"},{"–","-"},{" ,",","},{"“","\""},{"”","\""},{"‘","'"},{"’","'"}};
        for(String[] pair:replacements) text=text.replace(pair[0],pair[1]);
        char end=text.charAt(text.length()-1);
        if(".!?-,".indexOf(end)<0)text+=".";
        return text;
    }

    long[] encode(String text) {
        String input=normalize(text).replace(" ","[SPACE]");
        List<Long> ids=new ArrayList<>(); ids.add(255L);
        Matcher matcher=specialPattern.matcher(input);int start=0;
        while(matcher.find()) {
            ordinary(input.substring(start,matcher.start()),ids);
            ids.add(specials.get(matcher.group()).longValue());start=matcher.end();
        }
        ordinary(input.substring(start),ids);ids.add(0L);
        return ids.stream().mapToLong(Long::longValue).toArray();
    }

    private void ordinary(String text,List<Long> ids) {
        Matcher matcher=WORDS.matcher(text);
        while(matcher.find()) {
            List<String> pieces=new ArrayList<>();
            matcher.group().codePoints().forEach(cp -> {String s=new String(Character.toChars(cp));pieces.add(vocabulary.containsKey(s)?s:"[UNK]");});
            while(pieces.size()>1) {
                int best=Integer.MAX_VALUE,index=-1;
                for(int i=0;i<pieces.size()-1;i++) {
                    int rank=ranks.getOrDefault(pieces.get(i)+"\0"+pieces.get(i+1),Integer.MAX_VALUE);
                    if(rank<best){best=rank;index=i;}
                }
                if(index<0)break;
                pieces.set(index,pieces.get(index)+pieces.remove(index+1));
            }
            for(String piece:pieces)ids.add(vocabulary.getOrDefault(piece,1).longValue());
        }
    }

    static List<String> split(String text) {
        List<String> chunks=new ArrayList<>();String current="";
        for(String word:text.strip().split("(?U)\\s+")) {
            int[] points=word.codePoints().toArray();
            for(int start=0;start<points.length;start+=140) {
                String part=new String(points,start,Math.min(140,points.length-start));
                String candidate=current.isEmpty()?part:current+" "+part;
                if(candidate.codePointCount(0,candidate.length())>140){chunks.add(current);current=part;}
                else current=candidate;
            }
        }
        if(!current.isEmpty())chunks.add(current);
        if(chunks.isEmpty())throw new IllegalArgumentException("Text is empty");
        return chunks;
    }
}
