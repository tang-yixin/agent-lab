package lab2;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * 数据清洗作业。
 *
 * 三类处置：fix（修复）/ dedup（去重）/ quarantine（隔离）。
 *
 * 键设计：Mapper 输出"业务键"（评分=user+movie+timestamp，用户=uid，电影=mid），
 * 前缀 R/U/M 区分类型。Reducer 对同键分组：
 *  - 值完全相同 -> 去重保留一条；
 *  - 值不同 -> 冲突，全部隔离。
 *
 * 输出（按类型分目录，便于评分作业按路径识别）：
 *  - <output>/ratings/    清洗后评分
 *  - <output>/users/      清洗后用户
 *  - <output>/movies/     清洗后电影
 *  - <output>/quarantine/ 隔离记录（行 + 原因）
 *  - <output>/summary.json 处置统计
 */
public class CleanJob extends Configured implements Tool {

    private static final String OUT_RATINGS = "ratings";
    private static final String OUT_USERS = "users";
    private static final String OUT_MOVIES = "movies";
    private static final String OUT_QUARANTINE = "quarantine";

    public enum CleanCounters {
        FIXED,
        DEDUP,
        QUARANTINE,
        CONFLICT
    }

    public static class CleanMapper extends Mapper<LongWritable, Text, Text, Text> {

        private MultipleOutputs<Text, Text> mos;
        private final Text keyOut = new Text();
        private final Text valOut = new Text();
        private Set<String> userIds = new HashSet<>();
        private Set<String> movieIds = new HashSet<>();

        @Override
        protected void setup(Context context) throws IOException {
            mos = new MultipleOutputs<>(context);
            Configuration conf = context.getConfiguration();
            String usersPath = conf.get("lab2.users.path");
            String moviesPath = conf.get("lab2.movies.path");
            if (usersPath != null) userIds = ReferenceTables.loadIds(conf, usersPath);
            if (moviesPath != null) movieIds = ReferenceTables.loadIds(conf, moviesPath);
        }

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String line = value.toString();
            if (line.trim().isEmpty()) return;
            String filename = ((FileSplit) context.getInputSplit()).getPath().getName();
            if (filename.contains("rating")) cleanRatings(line, context);
            else if (filename.contains("user")) cleanUsers(line, context);
            else if (filename.contains("movie")) cleanMovies(line, context);
        }

        private void cleanRatings(String original, Context context)
                throws IOException, InterruptedException {
            String normalized = normalizeSeparator(original);
            String[] parts = normalized.split("::", -1);
            if (parts.length == 5 && "EXTRA_FIELD".equals(parts[4])) {
                parts = Arrays.copyOf(parts, 4);
            }
            if (parts.length < 4) { quarantine(original, "字段缺失(评分)", context); return; }
            if (parts.length > 4) { quarantine(original, "字段多余(评分)", context); return; }

            String uid = parts[0];
            String mid = parts[1];
            String rating = parts[2];
            String tsStr = parts[3];

            if (!DataRules.isInt(uid) || !DataRules.isInt(mid)) {
                quarantine(original, "ID非整数", context); return;
            }
            if (!DataRules.isInt(rating)) { quarantine(original, "评分非整数", context); return; }
            int r = Integer.parseInt(rating);
            if (r < 1 || r > 5) { quarantine(original, "评分越界", context); return; }
            if (!DataRules.isLong(tsStr)) { quarantine(original, "时间戳非法", context); return; }

            long ts = Long.parseLong(tsStr);
            boolean fixed = false;
            if (ts > DataRules.TS_MS_THRESHOLD) { ts /= 1000; fixed = true; }
            if (!DataRules.validTimestamp(ts)) {
                quarantine(original, "时间戳超出覆盖范围", context); return;
            }
            if (!userIds.isEmpty() && !userIds.contains(uid)) {
                quarantine(original, "引用不存在的UserID", context); return;
            }
            if (!movieIds.isEmpty() && !movieIds.contains(mid)) {
                quarantine(original, "引用不存在的MovieID", context); return;
            }

            String cleaned = uid + "::" + mid + "::" + rating + "::" + ts;
            if (!cleaned.equals(original)) fixed = true;
            if (fixed) context.getCounter(CleanCounters.FIXED).increment(1);
            keyOut.set("R\u0001" + uid + "\u0001" + mid + "\u0001" + ts);
            valOut.set(cleaned);
            context.write(keyOut, valOut);
        }

        private void cleanUsers(String original, Context context)
                throws IOException, InterruptedException {
            String normalized = normalizeSeparator(original);
            String[] parts = normalized.split("::", -1);
            if (parts.length == 6 && "EXTRA_FIELD".equals(parts[5])) {
                parts = Arrays.copyOf(parts, 5);
            }
            if (parts.length < 5) { quarantine(original, "字段缺失(用户)", context); return; }
            if (parts.length > 5) { quarantine(original, "字段多余(用户)", context); return; }

            String uid = parts[0];
            String gender = parts[1];
            String age = parts[2];
            String occ = parts[3];
            String zip = parts[4];

            if (!DataRules.isInt(uid)) { quarantine(original, "UserID非整数", context); return; }
            if (!DataRules.GENDERS.contains(gender)) { quarantine(original, "Gender编码非法", context); return; }
            if (!DataRules.AGES.contains(age)) { quarantine(original, "Age编码非法", context); return; }
            if (!DataRules.OCCUPATIONS.contains(occ)) { quarantine(original, "Occupation编码非法", context); return; }
            if (zip.trim().isEmpty()) { quarantine(original, "Zip-code缺失", context); return; }

            String cleaned = uid + "::" + gender + "::" + age + "::" + occ + "::" + zip;
            if (!cleaned.equals(original)) context.getCounter(CleanCounters.FIXED).increment(1);
            keyOut.set("U\u0001" + uid);
            valOut.set(cleaned);
            context.write(keyOut, valOut);
        }

        private void cleanMovies(String original, Context context)
                throws IOException, InterruptedException {
            String normalized = normalizeSeparator(original);
            String[] parts = normalized.split("::", -1);
            if (parts.length == 4 && "EXTRA_FIELD".equals(parts[3])) {
                parts = Arrays.copyOf(parts, 3);
            }
            if (parts.length < 3) { quarantine(original, "字段缺失(电影)", context); return; }
            if (parts.length > 3) { quarantine(original, "字段多余(电影)", context); return; }

            String mid = parts[0];
            String title = parts[1];
            String genres = parts[2];

            if (!DataRules.isInt(mid)) { quarantine(original, "MovieID非整数", context); return; }
            if (title.trim().isEmpty()) { quarantine(original, "标题缺失", context); return; }
            if (!DataRules.hasYear(title)) { quarantine(original, "标题年份缺失或异常", context); return; }
            if (genres.trim().isEmpty()) { quarantine(original, "类型缺失", context); return; }
            for (String g : genres.split("\\|")) {
                if (!DataRules.GENRES.contains(g)) {
                    quarantine(original, "类型名称非法或未知", context); return;
                }
            }

            String cleaned = mid + "::" + title + "::" + genres;
            if (!cleaned.equals(original)) context.getCounter(CleanCounters.FIXED).increment(1);
            keyOut.set("M\u0001" + mid);
            valOut.set(cleaned);
            context.write(keyOut, valOut);
        }

        private String normalizeSeparator(String line) {
            if (line.contains("::")) return line;
            if (line.contains(",")) return line.replace(",", "::");
            if (line.contains("|")) return line.replace("|", "::");
            if (line.contains(":")) return line.replace(":", "::");
            return line;
        }

        private void quarantine(String raw, String reason, Context context)
                throws IOException, InterruptedException {
            context.getCounter(CleanCounters.QUARANTINE).increment(1);
            keyOut.set(raw);
            valOut.set(reason);
            mos.write(OUT_QUARANTINE, keyOut, valOut, "quarantine/part");
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            if (mos != null) mos.close();
        }
    }

    public static class DedupReducer extends Reducer<Text, Text, NullWritable, Text> {

        private MultipleOutputs<NullWritable, Text> mos;
        private final NullWritable nk = NullWritable.get();

        @Override
        protected void setup(Context context) {
            mos = new MultipleOutputs<>(context);
        }

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            Set<String> distinct = new HashSet<>();
            int count = 0;
            for (Text v : values) {
                count++;
                distinct.add(v.toString());
            }

            char type = key.toString().charAt(0);
            String namedOutput;
            String basePath;
            switch (type) {
                case 'R': namedOutput = OUT_RATINGS; basePath = "ratings/part"; break;
                case 'U': namedOutput = OUT_USERS; basePath = "users/part"; break;
                case 'M': namedOutput = OUT_MOVIES; basePath = "movies/part"; break;
                default: return;
            }

            if (distinct.size() == 1) {
                if (count > 1) context.getCounter(CleanCounters.DEDUP).increment(count - 1);
                mos.write(namedOutput, nk, new Text(distinct.iterator().next()), basePath);
            } else {
                context.getCounter(CleanCounters.CONFLICT).increment(count);
                for (String s : distinct) {
                    mos.write(OUT_QUARANTINE, nk, new Text(s + "\t冲突(同键不同值)"), "quarantine/part");
                }
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            if (mos != null) mos.close();
        }
    }

    @Override
    public int run(String[] args) throws Exception {
        if (args.length < 4) {
            System.err.println("Usage: CleanJob <input> <output> <usersPath> <moviesPath>");
            return 1;
        }
        Path input = new Path(args[0]);
        Path output = new Path(args[1]);
        String usersPath = args[2];
        String moviesPath = args[3];

        Configuration conf = getConf();
        FileSystem fs = FileSystem.get(conf);
        if (fs.exists(output)) fs.delete(output, true);

        Job job = Job.getInstance(conf, "clean-job");
        job.setJarByClass(CleanJob.class);
        job.setMapperClass(CleanMapper.class);
        job.setReducerClass(DedupReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.getConfiguration().set("lab2.users.path", usersPath);
        job.getConfiguration().set("lab2.movies.path", moviesPath);
        FileInputFormat.addInputPath(job, input);
        FileOutputFormat.setOutputPath(job, output);

        MultipleOutputs.addNamedOutput(job, OUT_RATINGS,
                org.apache.hadoop.mapreduce.lib.output.TextOutputFormat.class,
                NullWritable.class, Text.class);
        MultipleOutputs.addNamedOutput(job, OUT_USERS,
                org.apache.hadoop.mapreduce.lib.output.TextOutputFormat.class,
                NullWritable.class, Text.class);
        MultipleOutputs.addNamedOutput(job, OUT_MOVIES,
                org.apache.hadoop.mapreduce.lib.output.TextOutputFormat.class,
                NullWritable.class, Text.class);
        MultipleOutputs.addNamedOutput(job, OUT_QUARANTINE,
                org.apache.hadoop.mapreduce.lib.output.TextOutputFormat.class,
                NullWritable.class, Text.class);

        boolean ok = job.waitForCompletion(true);
        if (!ok) return 1;

        long fixed = job.getCounters().findCounter(CleanCounters.FIXED).getValue();
        long dedup = job.getCounters().findCounter(CleanCounters.DEDUP).getValue();
        long quarantine = job.getCounters().findCounter(CleanCounters.QUARANTINE).getValue();
        long conflict = job.getCounters().findCounter(CleanCounters.CONFLICT).getValue();

        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"fixed\": ").append(fixed).append(",\n");
        sb.append("  \"dedup\": ").append(dedup).append(",\n");
        sb.append("  \"quarantine\": ").append(quarantine).append(",\n");
        sb.append("  \"conflict\": ").append(conflict).append(",\n");
        sb.append("  \"quarantine_total\": ").append(quarantine + conflict).append("\n");
        sb.append("}\n");

        Path summary = new Path(output, "summary.json");
        try (java.io.OutputStream os = fs.create(summary, true)) {
            os.write(sb.toString().getBytes("UTF-8"));
        }
        System.out.println("===== 清洗统计 =====");
        System.out.println(sb);
        return 0;
    }

    public static void main(String[] args) throws Exception {
        int exit = ToolRunner.run(new Configuration(), new CleanJob(), args);
        System.exit(exit);
    }
}
