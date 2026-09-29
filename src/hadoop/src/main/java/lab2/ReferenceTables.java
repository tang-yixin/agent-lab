package lab2;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashSet;
import java.util.Set;

/**
 * 加载 users.dat / movies.dat 中的合法 ID 集合，用于跨表关联检查（一致性）。
 * 采用 leadingId 提取，兼容脏分隔符，保证"存在性"判断稳健。
 */
public final class ReferenceTables {

    public static Set<String> loadIds(Configuration conf, String path) throws IOException {
        Set<String> ids = new HashSet<>();
        Path p = new Path(path);
        FileSystem fs = FileSystem.get(conf);
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(fs.open(p), "ISO-8859-1"))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                String id = DataRules.leadingId(line);
                if (id != null) {
                    ids.add(id);
                }
            }
        }
        return ids;
    }

    private ReferenceTables() {}
}
