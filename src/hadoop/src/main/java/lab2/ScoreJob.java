package lab2;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * 五维质量评分作业。
 *
 * 类型判定：根据文件名或父目录名中的关键词（rating/user/movie）判断记录类型。
 * 这样既能处理清洗前的 ratings.dat/users.dat/movies.dat，也能处理
 * 清洗后按 ratings/、users/、movies/ 子目录组织的输出。
 *
 * 输出 report.json：五维得分 + 各项计数。
 */
public class ScoreJob extends Configured implements Tool {

    public static final String TOTAL = "TOTAL";
    public static final String ACCURATE_OK = "ACCURATE_OK";
    public static final String COMPLETE_OK = "COMPLETE_OK";
    public static final String CONSISTENT_OK = "CONSISTENT_OK";
    public static final String UPTODATE_OK = "UPTODATE_OK";
    public static final String UPTODATE_TOTAL = "UPTODATE_TOTAL";
    public static final String DUP_TOTAL = "DUP_TOTAL";

    /** 根据文件路径判断记录类型：rating/user/movie。 */
    static String classifyType(Path path) {
        String name = path.getName().toLowerCase();
        String parent = path.getParent() != null ? path.getParent().getName().toLowerCase() : "";
        String s = name + "/" + parent;
        if (s.contains("rating")) return "RATINGS";
        if (s.contains("user")) return "USERS";
        if (s.contains("movie")) return "MOVIES";
        return "UNKNOWN";
    }

    public static class ScoreMapper extends Mapper<LongWritable, Text, Text, LongWritable> {

        private final Text outKey = new Text();
        private final LongWritable one = new LongWritable(1);
        private Set<String> userIds = new HashSet<>();
        private Set<String> movieIds = new HashSet<>();

        @Override
        protected void setup(Context context) throws java.io.IOException {
            Configuration conf = context.getConfiguration();
            String usersPath = conf.get("lab2.users.path");
            String moviesPath = conf.get("lab2.movies.path");
            if (usersPath != null) userIds = ReferenceTables.loadIds(conf, usersPath);
            if (moviesPath != null) movieIds = ReferenceTables.loadIds(conf, moviesPath);
        }

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws java.io.IOException, InterruptedException {
            String line = value.toString();
            if (line.trim().isEmpty()) return;

            String type = classifyType(((FileSplit) context.getInputSplit()).getPath());
            if ("UNKNOWN".equals(type)) return; // 跳过隔离目录等无关文件
            String[] parts = line.split("::", -1);

            emit(context, TOTAL, 1);

            if (isComplete(type, parts)) emit(context, COMPLETE_OK, 1);
            if (isConsistent(type, parts)) emit(context, CONSISTENT_OK, 1);
            if (isAccurate(type, parts)) emit(context, ACCURATE_OK, 1);

            if ("RATINGS".equals(type)) {
                emit(context, UPTODATE_TOTAL, 1);
                if (parts.length == 4 && DataRules.isLong(parts[3])) {
                    long ts = Long.parseLong(parts[3]);
                    if (ts > DataRules.TS_MS_THRESHOLD) ts /= 1000;
                    if (DataRules.validTimestamp(ts)) emit(context, UPTODATE_OK, 1);
                }
            }
        }

        private boolean isComplete(String type, String[] parts) {
            if ("RATINGS".equals(type)) return parts.length == 4 && DataRules.allNonEmpty(parts);
            if ("USERS".equals(type)) return parts.length == 5 && DataRules.allNonEmpty(parts);
            if ("MOVIES".equals(type)) return parts.length == 3 && DataRules.allNonEmpty(parts);
            return false;
        }

        private boolean isConsistent(String type, String[] parts) {
            if ("RATINGS".equals(type)) {
                if (parts.length != 4) return false;
                if (!userIds.isEmpty() && !movieIds.isEmpty()) {
                    String uid = DataRules.leadingId(parts[0]);
                    String mid = DataRules.leadingId(parts[1]);
                    return uid != null && mid != null && userIds.contains(uid) && movieIds.contains(mid);
                }
                return true;
            }
            if ("USERS".equals(type)) return parts.length == 5;
            if ("MOVIES".equals(type)) return parts.length == 3;
            return false;
        }

        private boolean isAccurate(String type, String[] parts) {
            if ("RATINGS".equals(type)) return DataRules.validRatingFields(parts);
            if ("USERS".equals(type)) return DataRules.validUserFields(parts);
            if ("MOVIES".equals(type)) return DataRules.validMovieFields(parts);
            return false;
        }

        private void emit(Context context, String k, long v) throws java.io.IOException, InterruptedException {
            outKey.set(k);
            one.set(v);
            context.write(outKey, one);
        }
    }

    public static class SumReducer extends Reducer<Text, LongWritable, Text, LongWritable> {
        private final LongWritable result = new LongWritable();

        @Override
        protected void reduce(Text key, Iterable<LongWritable> values, Context context)
                throws java.io.IOException, InterruptedException {
            long sum = 0;
            for (LongWritable v : values) sum += v.get();
            result.set(sum);
            context.write(key, result);
        }
    }

    public static class DupMapper extends Mapper<LongWritable, Text, Text, LongWritable> {
        private final Text outKey = new Text();
        private final LongWritable one = new LongWritable(1);

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws java.io.IOException, InterruptedException {
            String line = value.toString();
            if (line.trim().isEmpty()) return;
            String type = classifyType(((FileSplit) context.getInputSplit()).getPath());
            if ("UNKNOWN".equals(type)) return;
            outKey.set(type + "\u0001" + line);
            context.write(outKey, one);
        }
    }

    public static class DupReducer extends Reducer<Text, LongWritable, Text, LongWritable> {
        private final LongWritable dup = new LongWritable();

        @Override
        protected void reduce(Text key, Iterable<LongWritable> values, Context context)
                throws java.io.IOException, InterruptedException {
            long count = 0;
            for (LongWritable v : values) count += v.get();
            if (count > 1) {
                dup.set(count - 1);
                context.write(key, dup);
            }
        }
    }

    @Override
    public int run(String[] args) throws Exception {
        if (args.length < 4) {
            System.err.println("Usage: ScoreJob <input> <output> <usersPath> <moviesPath>");
            return 1;
        }
        Path input = new Path(args[0]);
        Path output = new Path(args[1]);
        String usersPath = args[2];
        String moviesPath = args[3];

        Configuration conf = getConf();
        FileSystem fs = FileSystem.get(conf);
        if (fs.exists(output)) fs.delete(output, true);

        Job job = Job.getInstance(conf, "score-count");
        job.setJarByClass(ScoreJob.class);
        job.setMapperClass(ScoreMapper.class);
        job.setReducerClass(SumReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(LongWritable.class);
        job.getConfiguration().set("lab2.users.path", usersPath);
        job.getConfiguration().setBoolean("mapreduce.input.fileinputformat.input.dir.recursive", true);
        job.getConfiguration().set("lab2.movies.path", moviesPath);
        FileInputFormat.addInputPath(job, input);
        Path countOut = new Path(output, "counts");
        FileOutputFormat.setOutputPath(job, countOut);
        if (!job.waitForCompletion(true)) return 1;

        Map<String, Long> counts = new HashMap<>();
        for (FileStatus st : fs.listStatus(countOut)) {
            if (!st.getPath().getName().startsWith("part-")) continue;
            try (BufferedReader br = new BufferedReader(
                    new InputStreamReader(fs.open(st.getPath()), "UTF-8"))) {
                String line;
                while ((line = br.readLine()) != null) {
                    String[] kv = line.split("\\t");
                    if (kv.length == 2) {
                        counts.put(kv[0], counts.getOrDefault(kv[0], 0L) + Long.parseLong(kv[1]));
                    }
                }
            }
        }

        Job dupJob = Job.getInstance(conf, "score-dedup");
        dupJob.setJarByClass(ScoreJob.class);
        dupJob.setMapperClass(DupMapper.class);
        dupJob.setReducerClass(DupReducer.class);
        dupJob.setOutputKeyClass(Text.class);
        dupJob.getConfiguration().setBoolean("mapreduce.input.fileinputformat.input.dir.recursive", true);
        dupJob.setOutputValueClass(LongWritable.class);
        FileInputFormat.addInputPath(dupJob, input);
        Path dupOut = new Path(output, "dups");
        FileOutputFormat.setOutputPath(dupJob, dupOut);
        if (!dupJob.waitForCompletion(true)) return 1;

        long dupTotal = 0;
        for (FileStatus st : fs.listStatus(dupOut)) {
            if (!st.getPath().getName().startsWith("part-")) continue;
            try (BufferedReader br = new BufferedReader(
                    new InputStreamReader(fs.open(st.getPath()), "UTF-8"))) {
                String line;
                while ((line = br.readLine()) != null) {
                    String[] kv = line.split("\\t");
                    if (kv.length == 2) dupTotal += Long.parseLong(kv[1]);
                }
            }
        }
        counts.put(DUP_TOTAL, dupTotal);

        long total = counts.getOrDefault(TOTAL, 0L);
        long accurate = counts.getOrDefault(ACCURATE_OK, 0L);
        long complete = counts.getOrDefault(COMPLETE_OK, 0L);
        long consistent = counts.getOrDefault(CONSISTENT_OK, 0L);
        long uptodate = counts.getOrDefault(UPTODATE_OK, 0L);
        long uptodateTotal = counts.getOrDefault(UPTODATE_TOTAL, 0L);
        long dup = counts.getOrDefault(DUP_TOTAL, 0L);

        double sAccurate = pct(accurate, total);
        double sComplete = pct(complete, total);
        double sUnique = pct(total - dup, total);
        double sUpToDate = pct(uptodate, uptodateTotal);
        double sConsistent = pct(consistent, total);
        double overall = 0.25 * sAccurate + 0.25 * sComplete
                + 0.20 * sUnique + 0.20 * sConsistent + 0.10 * sUpToDate;

        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"scores\": {\n");
        sb.append("    \"Accurate\": ").append(round2(sAccurate)).append(",\n");
        sb.append("    \"Complete\": ").append(round2(sComplete)).append(",\n");
        sb.append("    \"Unique\": ").append(round2(sUnique)).append(",\n");
        sb.append("    \"Up-to-date\": ").append(round2(sUpToDate)).append(",\n");
        sb.append("    \"Consistent\": ").append(round2(sConsistent)).append(",\n");
        sb.append("    \"Overall\": ").append(round2(overall)).append("\n");
        sb.append("  },\n");
        sb.append("  \"counts\": {\n");
        sb.append("    \"total\": ").append(total).append(",\n");
        sb.append("    \"accurate_ok\": ").append(accurate).append(",\n");
        sb.append("    \"complete_ok\": ").append(complete).append(",\n");
        sb.append("    \"consistent_ok\": ").append(consistent).append(",\n");
        sb.append("    \"uptodate_ok\": ").append(uptodate).append(",\n");
        sb.append("    \"uptodate_total\": ").append(uptodateTotal).append(",\n");
        sb.append("    \"duplicates\": ").append(dup).append("\n");
        sb.append("  }\n");
        sb.append("}\n");

        Path report = new Path(output, "report.json");
        try (java.io.OutputStream os = fs.create(report, true)) {
            os.write(sb.toString().getBytes("UTF-8"));
        }
        System.out.println("===== 五维评分报告 =====");
        System.out.println(sb);
        System.out.println("报告已写入: " + report);
        return 0;
    }

    private static double pct(long ok, long total) {
        return total == 0 ? 0.0 : 100.0 * ok / total;
    }

    private static double round2(double v) {
        return Math.round(v * 100.0) / 100.0;
    }

    public static void main(String[] args) throws Exception {
        int exit = ToolRunner.run(new Configuration(), new ScoreJob(), args);
        System.exit(exit);
    }
}
