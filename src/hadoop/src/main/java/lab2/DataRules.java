package lab2;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 数据治理共享规则：编码集、时间戳范围、T1/T2、字段判定。
 * 评分作业（ScoreJob）与清洗作业（CleanJob）共用本类，
 * 保证"清洗前后使用同一套评价口径"。
 */
public final class DataRules {

    public static final String SEP = "::";

    // ---- 合法编码集（依据 ml-1m README）----
    public static final Set<String> GENDERS = new HashSet<>(Arrays.asList("M", "F"));
    public static final Set<String> AGES = new HashSet<>(
            Arrays.asList("1", "18", "25", "35", "45", "50", "56"));
    public static final Set<String> OCCUPATIONS = new HashSet<>();
    static {
        for (int i = 0; i <= 20; i++) {
            OCCUPATIONS.add(String.valueOf(i));
        }
    }
    public static final Set<String> GENRES = new HashSet<>(Arrays.asList(
            "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
            "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
            "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"));

    // ---- 时间戳合法范围（秒级，对应数据集覆盖 2000-04 ~ 2003-02）----
    public static final long TS_MIN = 956_000_000L;
    public static final long TS_MAX = 1_048_000_000L;

    // 毫秒级时间戳判断阈值：超过该值视为毫秒，需 /1000
    public static final long TS_MS_THRESHOLD = 1_000_000_000_000L;

    // ---- 时间边界 T1/T2（迭代一定稿，供后续迭代使用）----
    public static final long T1 = 978_307_199L; // 2000-12-31 23:59:59 UTC
    public static final long T2 = 1_009_843_199L; // 2001-12-31 23:59:59 UTC

    // 记录类型
    public enum Type { RATINGS, USERS, MOVIES, UNKNOWN }

    private static final Pattern LEAD_ID = Pattern.compile("^(\\d+)");
    private static final Pattern YEAR = Pattern.compile("\\((\\d{4})\\)");

    /** 提取行首的整型 ID（用于跨表存在性检查，兼容脏分隔符）。 */
    public static String leadingId(String s) {
        Matcher m = LEAD_ID.matcher(s);
        return m.find() ? m.group(1) : null;
    }

    /** 按 "::" 切分字段数粗分类。字段数异常归为 UNKNOWN。 */
    public static Type classifyByParts(String[] parts) {
        switch (parts.length) {
            case 4: return Type.RATINGS;
            case 5: return Type.USERS;
            case 3: return Type.MOVIES;
            default: return Type.UNKNOWN;
        }
    }

    public static boolean isInt(String s) {
        if (s == null || s.isEmpty()) return false;
        for (int i = 0; i < s.length(); i++) {
            if (!Character.isDigit(s.charAt(i))) return false;
        }
        return true;
    }

    public static boolean isLong(String s) {
        if (s == null || s.isEmpty()) return false;
        try {
            Long.parseLong(s);
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }

    /** 字段是否全部非空（完整性用）。 */
    public static boolean allNonEmpty(String[] parts) {
        for (String p : parts) {
            if (p == null || p.trim().isEmpty()) return false;
        }
        return true;
    }

    /** 评分字段合法性（准确性用）：UserID/MovieID 整数、Rating 为 1-5 整数、Timestamp 整数。 */
    public static boolean validRatingFields(String[] parts) {
        if (parts.length != 4) return false;
        if (!isInt(parts[0]) || !isInt(parts[1]) || !isInt(parts[2]) || !isLong(parts[3])) {
            return false;
        }
        int r = Integer.parseInt(parts[2]);
        return r >= 1 && r <= 5;
    }

    /** 用户字段合法性：Gender/Age/Occupation 落在编码集。 */
    public static boolean validUserFields(String[] parts) {
        if (parts.length != 5) return false;
        if (!isInt(parts[0])) return false;
        return GENDERS.contains(parts[1])
                && AGES.contains(parts[2])
                && OCCUPATIONS.contains(parts[3]);
    }

    /** 标题是否包含合法年份（如 "(1995)"）。 */
    public static boolean hasYear(String title) {
        if (title == null) return false;
        Matcher m = YEAR.matcher(title);
        if (!m.find()) return false;
        int y = Integer.parseInt(m.group(1));
        return y >= 1880 && y <= 2010;
    }

    /** 电影字段合法性：MovieID 整数、标题非空且含合法年份、类型非空且均在合法类型集。 */
    public static boolean validMovieFields(String[] parts) {
        if (parts.length != 3) return false;
        if (!isInt(parts[0])) return false;
        if (parts[1] == null || parts[1].trim().isEmpty()) return false;
        if (!hasYear(parts[1])) return false;
        if (parts[2] == null || parts[2].trim().isEmpty()) return false;
        for (String g : parts[2].split("\\|")) {
            if (!GENRES.contains(g)) return false;
        }
        return true;
    }

    /** 时间戳是否落在数据集覆盖范围（时效性用）。 */
    public static boolean validTimestamp(long ts) {
        return ts >= TS_MIN && ts <= TS_MAX;
    }

    private DataRules() {}
}
