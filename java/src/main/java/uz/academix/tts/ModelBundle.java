package uz.academix.tts;

import com.fasterxml.jackson.databind.*;
import java.io.*;
import java.nio.file.*;
import java.security.*;
import java.util.HexFormat;

final class ModelBundle {
    private static final ObjectMapper JSON=new ObjectMapper();
    static Path resolve(Path modelDir,Path cache) throws Exception {
        if(modelDir!=null)return modelDir.toAbsolutePath();
        byte[] index;
        try(InputStream in=resource("bundle-index.json")){index=in.readAllBytes();}
        String id=HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(index)).substring(0,16);
        Path root=(cache!=null?cache:Path.of(System.getProperty("user.home"),".cache","uzbek-tts")).resolve(id).toAbsolutePath().normalize();
        Files.createDirectories(root);
        for(JsonNode entry:JSON.readTree(index).path("files")) {
            String name=entry.path("name").asText();Path output=root.resolve(name).normalize();
            if(!output.startsWith(root))throw new IOException("Invalid model resource path");
            long size=entry.path("size").asLong();
            if(Files.isRegularFile(output)&&Files.size(output)==size)continue;
            Path temp=Files.createTempFile(root,"model-",".tmp");
            try {
                MessageDigest digest=MessageDigest.getInstance("SHA-256");
                try(InputStream in=new DigestInputStream(resource(name),digest)) {Files.copy(in,temp,StandardCopyOption.REPLACE_EXISTING);}
                if(Files.size(temp)!=size || !HexFormat.of().formatHex(digest.digest()).equals(entry.path("sha256").asText()))
                    throw new IOException("Model checksum mismatch: "+name);
                Files.move(temp,output,StandardCopyOption.REPLACE_EXISTING,StandardCopyOption.ATOMIC_MOVE);
            } finally {Files.deleteIfExists(temp);}
        }
        return root;
    }
    private static InputStream resource(String name) throws IOException {
        InputStream input=ModelBundle.class.getResourceAsStream("/models/"+name);
        if(input==null)throw new FileNotFoundException("Model is missing from the JAR: "+name);
        return input;
    }
}
