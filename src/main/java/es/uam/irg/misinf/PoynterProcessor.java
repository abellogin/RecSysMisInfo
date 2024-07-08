package es.uam.irg.misinf;

import cern.colt.Arrays;
import com.opencsv.bean.CsvBindByName;
import com.opencsv.bean.CsvBindByPosition;
import com.opencsv.bean.CsvToBean;
import com.opencsv.bean.CsvToBeanBuilder;
import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.core.preference.PreferenceData;
import es.uam.eps.ir.ranksys.core.preference.SimplePreferenceData;
import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.FastUserIndex;
import es.uam.eps.ir.ranksys.fast.index.SimpleFastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.SimpleFastUserIndex;
import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import es.uam.eps.ir.ranksys.metrics.RecommendationMetric;
import es.uam.eps.ir.ranksys.metrics.SystemMetric;
import es.uam.eps.ir.ranksys.metrics.basic.AverageRecommendationMetric;
import es.uam.eps.ir.ranksys.mf.Factorization;
import es.uam.eps.ir.ranksys.mf.als.HKVFactorizer;
import es.uam.eps.ir.ranksys.mf.rec.MFRecommender;
import es.uam.eps.ir.ranksys.nn.item.ItemNeighborhoodRecommender;
import es.uam.eps.ir.ranksys.nn.item.neighborhood.ItemNeighborhood;
import es.uam.eps.ir.ranksys.nn.item.neighborhood.ItemNeighborhoods;
import es.uam.eps.ir.ranksys.nn.item.sim.ItemSimilarities;
import es.uam.eps.ir.ranksys.nn.item.sim.ItemSimilarity;
import es.uam.eps.ir.ranksys.nn.user.UserNeighborhoodRecommender;
import es.uam.eps.ir.ranksys.nn.user.neighborhood.UserNeighborhood;
import es.uam.eps.ir.ranksys.nn.user.neighborhood.UserNeighborhoods;
import es.uam.eps.ir.ranksys.nn.user.sim.UserSimilarities;
import es.uam.eps.ir.ranksys.nn.user.sim.UserSimilarity;
import es.uam.eps.ir.ranksys.rec.Recommender;
import es.uam.eps.ir.ranksys.rec.fast.basic.PopularityRecommender;
import es.uam.eps.ir.ranksys.rec.fast.basic.RandomRecommender;
import es.uam.eps.ir.ranksys.rec.runner.RecommenderRunner;
import es.uam.eps.ir.ranksys.rec.runner.fast.FastFilterRecommenderRunner;
import es.uam.eps.ir.ranksys.rec.runner.fast.FastFilters;
import es.uam.irg.misinf.eval.MisinformationCount;
import es.uam.irg.misinf.eval.MisinformationGini;
import es.uam.irg.misinf.eval.MisinformationRatioDelta;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Function;
import java.util.function.IntPredicate;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import org.jooq.lambda.Unchecked;
import org.ranksys.core.util.tuples.Tuple2od;
import static org.ranksys.formats.parsing.Parsers.sp;
import org.ranksys.formats.preference.SimpleRatingPreferencesReader;
import org.ranksys.formats.rec.RecommendationFormat;
import org.ranksys.formats.rec.SimpleRecommendationFormat;

/**
 *
 * @author Alejandro Bellogin
 */
public class PoynterProcessor {

    private CompletePoynterDataset completeData;
    private SimplePoynterDataset simpleData;
    private Set<String> misinformativeItems;
    private Set<String> usersWithMisinformativeItems;

    public PoynterProcessor(CompletePoynterDataset data) {
        completeData = data;
        simpleData = null;
    }

    public PoynterProcessor(SimplePoynterDataset data) {
        completeData = null;
        simpleData = data;
    }

    public Set<String> getMisinformativeItems() {
        if (misinformativeItems == null) {
            misinformativeItems = new HashSet<>();
            if (simpleData != null) {
                Map<String, String> equivalence = getEquivalentTweets(simpleData.getTweets());
                for (SimpleTweetBean t : simpleData.getTweets()) {
                    if (!t.getClaimId().isEmpty()) {
                        misinformativeItems.add(equivalence.getOrDefault(t.getId(), t.getId()));
                    }
                }
            } else if (completeData != null) {
                //Map<String, String> equivalence = getEquivalentTweets(completeData.getTweets());
                for (TweetBean t : completeData.getTweets()) {
                    if (!t.getClaimId().isEmpty()) {
                        misinformativeItems.add(t.getId());
                    }
                }
            }
        }
        return misinformativeItems;
    }

    public Set<String> getUsersWithMisinformativeItems() {
        if (usersWithMisinformativeItems == null) {
            usersWithMisinformativeItems = new HashSet<>();
            if (simpleData != null) {
                for (SimpleTweetBean t : simpleData.getTweets()) {
                    if (!t.getClaimId().isEmpty()) {
                        usersWithMisinformativeItems.add(t.getUserId());
                    }
                }
            } else if (completeData != null) {
                //Map<String, String> equivalence = getEquivalentTweets(completeData.getTweets());
                for (TweetBean t : completeData.getTweets()) {
                    if (!t.getClaimId().isEmpty()) {
                        usersWithMisinformativeItems.add(t.getUserId());
                    }
                }
            }
        }
        return usersWithMisinformativeItems;
    }

    private static Map<String, String> getEquivalentTweets(List<SimpleTweetBean> tweets) {
        Map<String, String> map = new HashMap<>();
        // first pass: find RTs
        for (SimpleTweetBean t : tweets) {
            if (!t.getParentId().isEmpty()) {
                map.put(t.getId(), t.getParentId());
            }
        }
        // second pass: find claims
        for (SimpleTweetBean t : tweets) {
            if (!t.getClaimId().isEmpty()) {
                if (map.containsKey(t.getId())) {
                    map.put(map.get(t.getId()), "claim_" + t.getClaimId());
                }
                map.put(t.getId(), "claim_" + t.getClaimId());
            }
        }
        return map;
    }

    public void generateUserItemFile(String outputFile, double misinformationPercentage) throws FileNotFoundException {
        //List<TweetBean> tweets = data.getTweets();
        List<SimpleTweetBean> tweets = simpleData.getTweets();
        // do one pass to merge tweet_ids with the same claim and/or those that are retweets (but not quotes)
        Map<String, String> equivalence = getEquivalentTweets(tweets);
        Map<String, Set<String>> positivePerUser = new HashMap<>();
        Map<String, Set<String>> negativePerUser = new HashMap<>();
        for (SimpleTweetBean tweet : tweets) {
            String u = tweet.getUserId();
            String i = tweet.getId();
            boolean isPos = tweet.getClaimId().isEmpty();
            if (isPos) {
                Set<String> items = positivePerUser.get(u);
                if (items == null) {
                    items = new HashSet<>();
                    positivePerUser.put(u, items);
                }
                items.add(i);
            } else {
                Set<String> items = negativePerUser.get(u);
                if (items == null) {
                    items = new HashSet<>();
                    negativePerUser.put(u, items);
                }
                items.add(i);
            }
        }
        // 
        PrintStream out = new PrintStream(outputFile);
        for (String u : positivePerUser.keySet()) {
            if (!negativePerUser.containsKey(u)) {
                System.out.println("User " + u + " does not include negative items");
                continue;
            }
            List<String> pos = new ArrayList<>(positivePerUser.get(u));
            int p = pos.size();
            List<String> neg = new ArrayList<>(negativePerUser.get(u));
            int n = neg.size();
            int t = p + n;
            if (Double.isNaN(misinformationPercentage)) {
                // consider the entire profile
                pos.forEach(i -> out.println(u + "\t" + equivalence.getOrDefault(i, i) + "\t1"));
                neg.forEach(i -> out.println(u + "\t" + equivalence.getOrDefault(i, i) + "\t1"));
            } else {
                int desNeg = (int) Math.floor(misinformationPercentage * 1.0 * t);
                int desPos = (int) Math.floor((1.0 - misinformationPercentage) * 1.0 * t);
                while ((desNeg > n) || (desPos > p)) {
                    if (desNeg > n) {
                        desNeg--;
                    }
                    if (desPos > p) {
                        desPos--;
                    }
                    int newTotal = desNeg + desPos;
                    desNeg = (int) Math.floor(misinformationPercentage * 1.0 * newTotal);
                    desPos = (int) Math.floor((1.0 - misinformationPercentage) * 1.0 * newTotal);
                }
                if (desNeg < n) {
                    // sample
                    Collections.shuffle(neg);
                    neg.subList(0, (int) desNeg).forEach(i -> out.println(u + "\t" + equivalence.getOrDefault(i, i) + "\t1"));
                } else {
                    neg.forEach(i -> out.println(u + "\t" + equivalence.getOrDefault(i, i) + "\t1"));
                }
                if (desPos < p) {
                    // sample
                    Collections.shuffle(pos);
                    pos.subList(0, (int) desPos).forEach(i -> out.println(u + "\t" + equivalence.getOrDefault(i, i) + "\t1"));
                } else {
                    pos.forEach(i -> out.println(u + "\t" + equivalence.getOrDefault(i, i) + "\t1"));
                }
            }
        }
        out.close();
    }

    public void generateUserFeatures(String outputFile) {

    }

    public void generateItemFeatures(String outputFile) {

    }

    public static class SimplePoynterDataset {

        private final String tweetFile;
        private final String userFile;
        private List<SimpleTweetBean> tweets;
        private List<AggregatedUserStats> users;

        public SimplePoynterDataset(String tweetFile, String userFile) {
            this.tweetFile = tweetFile;
            this.userFile = userFile;
        }

        public List<SimpleTweetBean> getTweets() {
            if (tweets == null) {
                tweets = CompletePoynterDataset.read(tweetFile, SimpleTweetBean.class);
            }
            System.out.println(tweets.size());
            return tweets;
        }

        public List<AggregatedUserStats> getUsers() {
            if (users == null) {
                users = CompletePoynterDataset.read(userFile, AggregatedUserStats.class);
            }
            System.out.println(users.size());
            return users;
        }

    }

    public static class CompletePoynterDataset {

        private final String claimFile;
        private final String claimSourceFile;
        private final String tweetFile;
        private final String userFile;
        private List<ClaimBean> claims;
        private List<ClaimSourceBean> claimSources;
        private List<TweetBean> tweets;
        private List<UserBean> users;

        public CompletePoynterDataset(String claimFile, String claimSourceFile, String tweetFile, String userFile) {
            this.claimFile = claimFile;
            this.claimSourceFile = claimSourceFile;
            this.tweetFile = tweetFile;
            this.userFile = userFile;
        }

        private static <T> List<T> read(String file, Class<T> c) {
            try {
                FileReader filereader = new FileReader(file);
                // http://zetcode.com/java/opencsv/
                CsvToBean<T> parser = new CsvToBeanBuilder<T>(filereader)
                        //.withMappingStrategy(strat)
                        //.withIgnoreLeadingWhiteSpace(true)
                        .withType(c)
                        .withMultilineLimit(1)
                        .build();
                //
                List<T> info = parser.parse();
                //info.forEach(System.out::println);
                return info;
            } catch (Exception e) {
                e.printStackTrace();
            }
            return null;
        }

        public List<ClaimBean> getClaims() {
            if (claims == null) {
                claims = read(claimFile, ClaimBean.class);
            }
            return claims;
        }

        public List<ClaimSourceBean> getClaimSources() {
            if (claimSources == null) {
                claimSources = read(claimSourceFile, ClaimSourceBean.class);
            }
            return claimSources;
        }

        public List<TweetBean> getTweets() {
            if (tweets == null) {
                tweets = read(tweetFile, TweetBean.class);
            }
            return tweets;
        }

        public List<UserBean> getUsers() {
            if (users == null) {
                users = read(userFile, UserBean.class);
            }
            return users;
        }

        public void readClaims() {
            read(claimFile, ClaimBean.class);
        }

        public void readClaimSources() {
            read(claimSourceFile, ClaimSourceBean.class);
        }

        public void readTweets() {
            read(tweetFile, TweetBean.class);
        }

        public void readUsers() {
            read(userFile, UserBean.class);
        }
    }

    public static class ClaimBean {

        @CsvBindByPosition(position = 0)
        private String id;

        public String getId() {
            return id;
        }

        public void setId(String id) {
            this.id = id;
        }

        @Override
        public String toString() {
            return "ClaimBean{" + "id=" + id + '}';
        }

    }

    public static class ClaimSourceBean {

        @CsvBindByPosition(position = 0)
        private String id;
        @CsvBindByPosition(position = 1)
        private String claimId;
        @CsvBindByPosition(position = 2)
        private String content;
        @CsvBindByPosition(position = 3)
        private String url;
        @CsvBindByPosition(position = 4)
        private String tweetId;
        @CsvBindByPosition(position = 5)
        private String source;
        @CsvBindByPosition(position = 6)
        private int sourceId;

        public String getId() {
            return id;
        }

        public void setId(String id) {
            this.id = id;
        }

        public String getClaimId() {
            return claimId;
        }

        public void setClaimId(String claimId) {
            this.claimId = claimId;
        }

        public String getContent() {
            return content;
        }

        public void setContent(String content) {
            this.content = content;
        }

        public String getUrl() {
            return url;
        }

        public void setUrl(String url) {
            this.url = url;
        }

        public String getTweetId() {
            return tweetId;
        }

        public void setTweetId(String tweetId) {
            this.tweetId = tweetId;
        }

        public String getSource() {
            return source;
        }

        public void setSource(String source) {
            this.source = source;
        }

        public int getSourceId() {
            return sourceId;
        }

        public void setSourceId(int sourceId) {
            this.sourceId = sourceId;
        }

    }

    public static class AggregatedUserStats {

        @CsvBindByPosition(position = 0)
        private String userId;
        @CsvBindByPosition(position = 1)
        private int numTweets;
        @CsvBindByPosition(position = 2)
        private int numClaims;

        public String getUserId() {
            return userId;
        }

        public void setUserId(String userId) {
            this.userId = userId;
        }

        public int getNumTweets() {
            return numTweets;
        }

        public void setNumTweets(int numTweets) {
            this.numTweets = numTweets;
        }

        public int getNumClaims() {
            return numClaims;
        }

        public void setNumClaims(int numClaims) {
            this.numClaims = numClaims;
        }

    }

    public static class SimpleTweetBeanOld {

        @CsvBindByPosition(position = 0)
        private String userId;
        @CsvBindByPosition(position = 1)
        private String id;
        @CsvBindByPosition(position = 2)
        private String claimId;

        public String getUserId() {
            return userId;
        }

        public void setUserId(String userId) {
            this.userId = userId;
        }

        public String getId() {
            return id;
        }

        public void setId(String id) {
            this.id = id;
        }

        public String getClaimId() {
            return claimId;
        }

        public void setClaimId(String claimId) {
            this.claimId = claimId;
        }

    }

    public static class SimpleTweetBean {

        @CsvBindByName(column = "user_id")
        private String userId;
        @CsvBindByName(column = "id")
        private String id;
        @CsvBindByName(column = "parent_id")
        private String parentId;
        @CsvBindByName(column = "claim_id")
        private String claimId;

        public void setParentId(String parentId) {
            this.parentId = parentId;
        }

        public String getParentId() {
            return parentId;
        }

        public String getUserId() {
            return userId;
        }

        public void setUserId(String userId) {
            this.userId = userId;
        }

        public String getId() {
            return id;
        }

        public void setId(String id) {
            this.id = id;
        }

        public String getClaimId() {
            return claimId;
        }

        public void setClaimId(String claimId) {
            this.claimId = claimId;
        }

    }

    public static class TweetBean {

        @CsvBindByPosition(position = 0)
        private String id;
        @CsvBindByPosition(position = 1)
        private String userId;
        @CsvBindByPosition(position = 2)
        private String parentId;
        @CsvBindByPosition(position = 3)
        private int isRetweet;
        @CsvBindByPosition(position = 4)
        private int isQuote;
        @CsvBindByPosition(position = 5)
        private String source;
        @CsvBindByPosition(position = 6)
        private String timestamp;
        @CsvBindByPosition(position = 7)
        private String content;
        @CsvBindByPosition(position = 8)
        private String lang;
        @CsvBindByPosition(position = 9)
        private int rtCount;
        @CsvBindByPosition(position = 10)
        private int favCount;
        @CsvBindByPosition(position = 11)
        private String tags;
        @CsvBindByPosition(position = 12)
        private String urls;
        @CsvBindByPosition(position = 13)
        private String expandedUrls;
        @CsvBindByPosition(position = 14)
        private String mentions;
        @CsvBindByPosition(position = 15)
        private int hasMedia;
        @CsvBindByPosition(position = 16)
        private String claimId;

        public String getId() {
            return id;
        }

        public void setId(String id) {
            this.id = id;
        }

        public String getUserId() {
            return userId;
        }

        public void setUserId(String userId) {
            this.userId = userId;
        }

        public String getParentId() {
            return parentId;
        }

        public void setParentId(String parentId) {
            this.parentId = parentId;
        }

        public int getIsRetweet() {
            return isRetweet;
        }

        public void setIsRetweet(int isRetweet) {
            this.isRetweet = isRetweet;
        }

        public int getIsQuote() {
            return isQuote;
        }

        public void setIsQuote(int isQuote) {
            this.isQuote = isQuote;
        }

        public String getSource() {
            return source;
        }

        public void setSource(String source) {
            this.source = source;
        }

        public String getTimestamp() {
            return timestamp;
        }

        public void setTimestamp(String timestamp) {
            this.timestamp = timestamp;
        }

        public String getContent() {
            return content;
        }

        public void setContent(String content) {
            this.content = content;
        }

        public String getLang() {
            return lang;
        }

        public void setLang(String lang) {
            this.lang = lang;
        }

        public int getRtCount() {
            return rtCount;
        }

        public void setRtCount(int rtCount) {
            this.rtCount = rtCount;
        }

        public int getFavCount() {
            return favCount;
        }

        public void setFavCount(int favCount) {
            this.favCount = favCount;
        }

        public String getTags() {
            return tags;
        }

        public void setTags(String tags) {
            this.tags = tags;
        }

        public String getUrls() {
            return urls;
        }

        public void setUrls(String urls) {
            this.urls = urls;
        }

        public String getExpandedUrls() {
            return expandedUrls;
        }

        public void setExpandedUrls(String expandedUrls) {
            this.expandedUrls = expandedUrls;
        }

        public String getMentions() {
            return mentions;
        }

        public void setMentions(String mentions) {
            this.mentions = mentions;
        }

        public int getHasMedia() {
            return hasMedia;
        }

        public void setHasMedia(int hasMedia) {
            this.hasMedia = hasMedia;
        }

        public String getClaimId() {
            return claimId;
        }

        public void setClaimId(String claimId) {
            this.claimId = claimId;
        }

    }

    public static class UserBean {

        @CsvBindByPosition(position = 0)
        private String id;
        @CsvBindByPosition(position = 1)
        private String screenName;
        @CsvBindByPosition(position = 2)
        private String email;
        @CsvBindByPosition(position = 3)
        private int tweetCount;
        @CsvBindByPosition(position = 4)
        private int followerCount;
        @CsvBindByPosition(position = 5)
        private int friendCount;
        @CsvBindByPosition(position = 6)
        private String description;
        @CsvBindByPosition(position = 7)
        private String location;
        @CsvBindByPosition(position = 8)
        private int verified;
        @CsvBindByPosition(position = 9)
        private String createdAt;
        @CsvBindByPosition(position = 10)
        private int workerGroup; // to be ignored
        @CsvBindByPosition(position = 11)
        private String processedAt; // to be ignored

        public String getId() {
            return id;
        }

        public void setId(String id) {
            this.id = id;
        }

        public String getScreenName() {
            return screenName;
        }

        public void setScreenName(String screenName) {
            this.screenName = screenName;
        }

        public String getEmail() {
            return email;
        }

        public void setEmail(String email) {
            this.email = email;
        }

        public int getTweetCount() {
            return tweetCount;
        }

        public void setTweetCount(int tweetCount) {
            this.tweetCount = tweetCount;
        }

        public int getFollowerCount() {
            return followerCount;
        }

        public void setFollowerCount(int followerCount) {
            this.followerCount = followerCount;
        }

        public int getFriendCount() {
            return friendCount;
        }

        public void setFriendCount(int friendCount) {
            this.friendCount = friendCount;
        }

        public String getDescription() {
            return description;
        }

        public void setDescription(String description) {
            this.description = description;
        }

        public String getLocation() {
            return location;
        }

        public void setLocation(String location) {
            this.location = location;
        }

        public int getVerified() {
            return verified;
        }

        public void setVerified(int verified) {
            this.verified = verified;
        }

        public String getCreatedAt() {
            return createdAt;
        }

        public void setCreatedAt(String createdAt) {
            this.createdAt = createdAt;
        }

        public int getWorkerGroup() {
            return workerGroup;
        }

        public void setWorkerGroup(int workerGroup) {
            this.workerGroup = workerGroup;
        }

        public String getProcessedAt() {
            return processedAt;
        }

        public void setProcessedAt(String processedAt) {
            this.processedAt = processedAt;
        }

    }

    private static Map<String, Supplier<Recommender<String, String>>> getRecommenders(FastUserIndex<String> userIndex, FastItemIndex<String> itemIndex, FastPreferenceData<String, String> training) {
        Map<String, Supplier<Recommender<String, String>>> recMap = new HashMap<>();

        recMap.put("pop", () -> new PopularityRecommender<>(training));

        recMap.put("rnd", () -> new RandomRecommender<>(training, training));
        //
        recMap.put("ub_10_1", () -> {
            int k = 10;
            int q = 1;
            UserSimilarity<String> userSim = UserSimilarities.vectorCosine(training, true);
            UserNeighborhood<String> userNeighborhood = UserNeighborhoods.topK(userSim, k);
            return new UserNeighborhoodRecommender<>(training, userNeighborhood, q);
        });
        recMap.put("ub_50_1", () -> {
            int k = 50;
            int q = 1;
            UserSimilarity<String> userSim = UserSimilarities.vectorCosine(training, true);
            UserNeighborhood<String> userNeighborhood = UserNeighborhoods.topK(userSim, k);
            return new UserNeighborhoodRecommender<>(training, userNeighborhood, q);
        });
        recMap.put("ub_50_2", () -> {
            int k = 50;
            int q = 2;
            UserSimilarity<String> userSim = UserSimilarities.vectorCosine(training, true);
            UserNeighborhood<String> userNeighborhood = UserNeighborhoods.topK(userSim, k);
            return new UserNeighborhoodRecommender<>(training, userNeighborhood, q);
        });
        recMap.put("ub_50_3", () -> {
            int k = 50;
            int q = 3;
            UserSimilarity<String> userSim = UserSimilarities.vectorCosine(training, true);
            UserNeighborhood<String> userNeighborhood = UserNeighborhoods.topK(userSim, k);
            return new UserNeighborhoodRecommender<>(training, userNeighborhood, q);
        });
        recMap.put("ub_50_1__setjac", () -> {
            int k = 50;
            int q = 1;
            UserSimilarity<String> userSim = UserSimilarities.setJaccard(training, true);
            UserNeighborhood<String> userNeighborhood = UserNeighborhoods.topK(userSim, k);
            return new UserNeighborhoodRecommender<>(training, userNeighborhood, q);
        });
        recMap.put("ub_50_2__setjac", () -> {
            int k = 50;
            int q = 2;
            UserSimilarity<String> userSim = UserSimilarities.setJaccard(training, true);
            UserNeighborhood<String> userNeighborhood = UserNeighborhoods.topK(userSim, k);
            return new UserNeighborhoodRecommender<>(training, userNeighborhood, q);
        });
        recMap.put("ub_50_3__setjac", () -> {
            int k = 50;
            int q = 3;
            UserSimilarity<String> userSim = UserSimilarities.setJaccard(training, true);
            UserNeighborhood<String> userNeighborhood = UserNeighborhoods.topK(userSim, k);
            return new UserNeighborhoodRecommender<>(training, userNeighborhood, q);
        });
        recMap.put("ub_100_1", () -> {
            int k = 100;
            int q = 1;
            UserSimilarity<String> userSim = UserSimilarities.vectorCosine(training, true);
            UserNeighborhood<String> userNeighborhood = UserNeighborhoods.topK(userSim, k);
            return new UserNeighborhoodRecommender<>(training, userNeighborhood, q);
        });
        recMap.put("ub_10_2", () -> {
            int k = 10;
            int q = 2;
            UserSimilarity<String> userSim = UserSimilarities.vectorCosine(training, true);
            UserNeighborhood<String> userNeighborhood = UserNeighborhoods.topK(userSim, k);
            return new UserNeighborhoodRecommender<>(training, userNeighborhood, q);
        });
        recMap.put("ub_10_3", () -> {
            int k = 10;
            int q = 3;
            UserSimilarity<String> userSim = UserSimilarities.vectorCosine(training, true);
            UserNeighborhood<String> userNeighborhood = UserNeighborhoods.topK(userSim, k);
            return new UserNeighborhoodRecommender<>(training, userNeighborhood, q);
        });
        recMap.put("ub_100_2", () -> {
            int k = 100;
            int q = 2;
            UserSimilarity<String> userSim = UserSimilarities.vectorCosine(training, true);
            UserNeighborhood<String> userNeighborhood = UserNeighborhoods.topK(userSim, k);
            return new UserNeighborhoodRecommender<>(training, userNeighborhood, q);
        });
        recMap.put("ub_100_3", () -> {
            int k = 100;
            int q = 3;
            UserSimilarity<String> userSim = UserSimilarities.vectorCosine(training, true);
            UserNeighborhood<String> userNeighborhood = UserNeighborhoods.topK(userSim, k);
            return new UserNeighborhoodRecommender<>(training, userNeighborhood, q);
        });
        recMap.put("ub_100_1__setjac", () -> {
            int k = 100;
            int q = 1;
            UserSimilarity<String> userSim = UserSimilarities.setJaccard(training, true);
            UserNeighborhood<String> userNeighborhood = UserNeighborhoods.topK(userSim, k);
            return new UserNeighborhoodRecommender<>(training, userNeighborhood, q);
        });
        recMap.put("ub_10_1__setjac", () -> {
            int k = 10;
            int q = 1;
            UserSimilarity<String> userSim = UserSimilarities.setJaccard(training, true);
            UserNeighborhood<String> userNeighborhood = UserNeighborhoods.topK(userSim, k);
            return new UserNeighborhoodRecommender<>(training, userNeighborhood, q);
        });
        recMap.put("ub_100_2__setjac", () -> {
            int k = 100;
            int q = 2;
            UserSimilarity<String> userSim = UserSimilarities.setJaccard(training, true);
            UserNeighborhood<String> userNeighborhood = UserNeighborhoods.topK(userSim, k);
            return new UserNeighborhoodRecommender<>(training, userNeighborhood, q);
        });
        recMap.put("ub_10_2__setjac", () -> {
            int k = 10;
            int q = 2;
            UserSimilarity<String> userSim = UserSimilarities.setJaccard(training, true);
            UserNeighborhood<String> userNeighborhood = UserNeighborhoods.topK(userSim, k);
            return new UserNeighborhoodRecommender<>(training, userNeighborhood, q);
        });
        recMap.put("ub_100_3__setjac", () -> {
            int k = 100;
            int q = 3;
            UserSimilarity<String> userSim = UserSimilarities.setJaccard(training, true);
            UserNeighborhood<String> userNeighborhood = UserNeighborhoods.topK(userSim, k);
            return new UserNeighborhoodRecommender<>(training, userNeighborhood, q);
        });
        recMap.put("ub_10_3__setjac", () -> {
            int k = 10;
            int q = 3;
            UserSimilarity<String> userSim = UserSimilarities.setJaccard(training, true);
            UserNeighborhood<String> userNeighborhood = UserNeighborhoods.topK(userSim, k);
            return new UserNeighborhoodRecommender<>(training, userNeighborhood, q);
        });
        recMap.put("ub_100_1__setcos_0.5", () -> {
            int k = 100;
            int q = 1;
            double alpha = 0.5;
            UserSimilarity<String> userSim = UserSimilarities.setCosine(training, alpha, true);
            UserNeighborhood<String> userNeighborhood = UserNeighborhoods.topK(userSim, k);
            return new UserNeighborhoodRecommender<>(training, userNeighborhood, q);
        });
        recMap.put("ub_10_2__setcos_0.5", () -> {
            int k = 10;
            int q = 2;
            double alpha = 0.5;
            UserSimilarity<String> userSim = UserSimilarities.setCosine(training, alpha, true);
            UserNeighborhood<String> userNeighborhood = UserNeighborhoods.topK(userSim, k);
            return new UserNeighborhoodRecommender<>(training, userNeighborhood, q);
        });
        recMap.put("ub_100_2__setcos_0.5", () -> {
            int k = 100;
            int q = 2;
            double alpha = 0.5;
            UserSimilarity<String> userSim = UserSimilarities.setCosine(training, alpha, true);
            UserNeighborhood<String> userNeighborhood = UserNeighborhoods.topK(userSim, k);
            return new UserNeighborhoodRecommender<>(training, userNeighborhood, q);
        });
        //
        recMap.put("ib_10_1", () -> {
            int k = 10;
            int q = 1;
            ItemSimilarity<String> itemSim = ItemSimilarities.vectorCosine(training, true);
            ItemNeighborhood<String> itemNeighborhood = ItemNeighborhoods.cached(ItemNeighborhoods.topK(itemSim, k));
            return new ItemNeighborhoodRecommender<>(training, itemNeighborhood, q);
        });
        recMap.put("ib_10_1__setjac", () -> {
            int k = 10;
            int q = 1;
            ItemSimilarity<String> itemSim = ItemSimilarities.setJaccard(training, true);
            ItemNeighborhood<String> itemNeighborhood = ItemNeighborhoods.cached(ItemNeighborhoods.topK(itemSim, k));
            return new ItemNeighborhoodRecommender<>(training, itemNeighborhood, q);
        });
        recMap.put("ib_10_2__setjac", () -> {
            int k = 10;
            int q = 2;
            ItemSimilarity<String> itemSim = ItemSimilarities.setJaccard(training, true);
            ItemNeighborhood<String> itemNeighborhood = ItemNeighborhoods.cached(ItemNeighborhoods.topK(itemSim, k));
            return new ItemNeighborhoodRecommender<>(training, itemNeighborhood, q);
        });
        recMap.put("ib_10_3__setjac", () -> {
            int k = 10;
            int q = 3;
            ItemSimilarity<String> itemSim = ItemSimilarities.setJaccard(training, true);
            ItemNeighborhood<String> itemNeighborhood = ItemNeighborhoods.cached(ItemNeighborhoods.topK(itemSim, k));
            return new ItemNeighborhoodRecommender<>(training, itemNeighborhood, q);
        });
        recMap.put("ib_50_1", () -> {
            int k = 50;
            int q = 1;
            ItemSimilarity<String> itemSim = ItemSimilarities.vectorCosine(training, true);
            ItemNeighborhood<String> itemNeighborhood = ItemNeighborhoods.cached(ItemNeighborhoods.topK(itemSim, k));
            return new ItemNeighborhoodRecommender<>(training, itemNeighborhood, q);
        });
        recMap.put("ib_50_1__setjac", () -> {
            int k = 50;
            int q = 1;
            ItemSimilarity<String> itemSim = ItemSimilarities.setJaccard(training, true);
            ItemNeighborhood<String> itemNeighborhood = ItemNeighborhoods.cached(ItemNeighborhoods.topK(itemSim, k));
            return new ItemNeighborhoodRecommender<>(training, itemNeighborhood, q);
        });
        recMap.put("ib_50_2__setjac", () -> {
            int k = 50;
            int q = 2;
            ItemSimilarity<String> itemSim = ItemSimilarities.setJaccard(training, true);
            ItemNeighborhood<String> itemNeighborhood = ItemNeighborhoods.cached(ItemNeighborhoods.topK(itemSim, k));
            return new ItemNeighborhoodRecommender<>(training, itemNeighborhood, q);
        });
        recMap.put("ib_50_3__setjac", () -> {
            int k = 50;
            int q = 3;
            ItemSimilarity<String> itemSim = ItemSimilarities.setJaccard(training, true);
            ItemNeighborhood<String> itemNeighborhood = ItemNeighborhoods.cached(ItemNeighborhoods.topK(itemSim, k));
            return new ItemNeighborhoodRecommender<>(training, itemNeighborhood, q);
        });
        recMap.put("ib_100_1", () -> {
            int k = 100;
            int q = 1;
            ItemSimilarity<String> itemSim = ItemSimilarities.vectorCosine(training, true);
            ItemNeighborhood<String> itemNeighborhood = ItemNeighborhoods.cached(ItemNeighborhoods.topK(itemSim, k));
            return new ItemNeighborhoodRecommender<>(training, itemNeighborhood, q);
        });
        recMap.put("ib_100_1__setjac", () -> {
            int k = 100;
            int q = 1;
            ItemSimilarity<String> itemSim = ItemSimilarities.setJaccard(training, true);
            ItemNeighborhood<String> itemNeighborhood = ItemNeighborhoods.cached(ItemNeighborhoods.topK(itemSim, k));
            return new ItemNeighborhoodRecommender<>(training, itemNeighborhood, q);
        });
        recMap.put("ib_100_2__setjac", () -> {
            int k = 100;
            int q = 2;
            ItemSimilarity<String> itemSim = ItemSimilarities.setJaccard(training, true);
            ItemNeighborhood<String> itemNeighborhood = ItemNeighborhoods.cached(ItemNeighborhoods.topK(itemSim, k));
            return new ItemNeighborhoodRecommender<>(training, itemNeighborhood, q);
        });
        recMap.put("ib_100_3__setjac", () -> {
            int k = 100;
            int q = 3;
            ItemSimilarity<String> itemSim = ItemSimilarities.setJaccard(training, true);
            ItemNeighborhood<String> itemNeighborhood = ItemNeighborhoods.cached(ItemNeighborhoods.topK(itemSim, k));
            return new ItemNeighborhoodRecommender<>(training, itemNeighborhood, q);
        });
        recMap.put("ib_100_1__setcos_0.5", () -> {
            int k = 100;
            int q = 1;
            double alpha = 0.5;
            ItemSimilarity<String> itemSim = ItemSimilarities.setCosine(training, alpha, true);
            ItemNeighborhood<String> itemNeighborhood = ItemNeighborhoods.cached(ItemNeighborhoods.topK(itemSim, k));
            return new ItemNeighborhoodRecommender<>(training, itemNeighborhood, q);
        });
        recMap.put("ib_10_1__setcos_0.5", () -> {
            int k = 10;
            int q = 1;
            double alpha = 0.5;
            ItemSimilarity<String> itemSim = ItemSimilarities.setCosine(training, alpha, true);
            ItemNeighborhood<String> itemNeighborhood = ItemNeighborhoods.cached(ItemNeighborhoods.topK(itemSim, k));
            return new ItemNeighborhoodRecommender<>(training, itemNeighborhood, q);
        });
        recMap.put("ib_50_1__setcos_0.5", () -> {
            int k = 50;
            int q = 1;
            double alpha = 0.5;
            ItemSimilarity<String> itemSim = ItemSimilarities.setCosine(training, alpha, true);
            ItemNeighborhood<String> itemNeighborhood = ItemNeighborhoods.cached(ItemNeighborhoods.topK(itemSim, k));
            return new ItemNeighborhoodRecommender<>(training, itemNeighborhood, q);
        });
        recMap.put("ib_10_2", () -> {
            int k = 10;
            int q = 2;
            ItemSimilarity<String> itemSim = ItemSimilarities.vectorCosine(training, true);
            ItemNeighborhood<String> itemNeighborhood = ItemNeighborhoods.cached(ItemNeighborhoods.topK(itemSim, k));
            return new ItemNeighborhoodRecommender<>(training, itemNeighborhood, q);
        });
        recMap.put("ib_50_2", () -> {
            int k = 50;
            int q = 2;
            ItemSimilarity<String> itemSim = ItemSimilarities.vectorCosine(training, true);
            ItemNeighborhood<String> itemNeighborhood = ItemNeighborhoods.cached(ItemNeighborhoods.topK(itemSim, k));
            return new ItemNeighborhoodRecommender<>(training, itemNeighborhood, q);
        });
        recMap.put("ib_100_2", () -> {
            int k = 100;
            int q = 2;
            ItemSimilarity<String> itemSim = ItemSimilarities.vectorCosine(training, true);
            ItemNeighborhood<String> itemNeighborhood = ItemNeighborhoods.cached(ItemNeighborhoods.topK(itemSim, k));
            return new ItemNeighborhoodRecommender<>(training, itemNeighborhood, q);
        });
        recMap.put("ib_10_3", () -> {
            int k = 10;
            int q = 3;
            ItemSimilarity<String> itemSim = ItemSimilarities.vectorCosine(training, true);
            ItemNeighborhood<String> itemNeighborhood = ItemNeighborhoods.cached(ItemNeighborhoods.topK(itemSim, k));
            return new ItemNeighborhoodRecommender<>(training, itemNeighborhood, q);
        });
        recMap.put("ib_50_3", () -> {
            int k = 50;
            int q = 3;
            ItemSimilarity<String> itemSim = ItemSimilarities.vectorCosine(training, true);
            ItemNeighborhood<String> itemNeighborhood = ItemNeighborhoods.cached(ItemNeighborhoods.topK(itemSim, k));
            return new ItemNeighborhoodRecommender<>(training, itemNeighborhood, q);
        });
        recMap.put("ib_100_3", () -> {
            int k = 100;
            int q = 3;
            ItemSimilarity<String> itemSim = ItemSimilarities.vectorCosine(training, true);
            ItemNeighborhood<String> itemNeighborhood = ItemNeighborhoods.cached(ItemNeighborhoods.topK(itemSim, k));
            return new ItemNeighborhoodRecommender<>(training, itemNeighborhood, q);
        });
        recMap.put("hkv_50_0.1_1.0_20", () -> {
            int k = 50;
            double lambda = 0.1;
            double alpha = 1.0;
            DoubleUnaryOperator confidence = x -> 1 + alpha * x;
            int numIter = 20;

            Factorization<String, String> factorization = new HKVFactorizer<String, String>(lambda, confidence, numIter).factorize(k, training);

            return new MFRecommender<>(userIndex, itemIndex, factorization);
        });
        recMap.put("hkv_50_0.1_1.0_100", () -> {
            int k = 50;
            double lambda = 0.1;
            double alpha = 1.0;
            DoubleUnaryOperator confidence = x -> 1 + alpha * x;
            int numIter = 100;

            Factorization<String, String> factorization = new HKVFactorizer<String, String>(lambda, confidence, numIter).factorize(k, training);

            return new MFRecommender<>(userIndex, itemIndex, factorization);
        });
        recMap.put("hkv_50_0.01_1.0_20", () -> {
            int k = 50;
            double lambda = 0.01;
            double alpha = 1.0;
            DoubleUnaryOperator confidence = x -> 1 + alpha * x;
            int numIter = 20;

            Factorization<String, String> factorization = new HKVFactorizer<String, String>(lambda, confidence, numIter).factorize(k, training);

            return new MFRecommender<>(userIndex, itemIndex, factorization);
        });
        recMap.put("hkv_50_0.1_1.0_100", () -> {
            int k = 50;
            double lambda = 0.01;
            double alpha = 1.0;
            DoubleUnaryOperator confidence = x -> 1 + alpha * x;
            int numIter = 100;

            Factorization<String, String> factorization = new HKVFactorizer<String, String>(lambda, confidence, numIter).factorize(k, training);

            return new MFRecommender<>(userIndex, itemIndex, factorization);
        });
        recMap.put("hkv_20_0.1_1.0_20", () -> {
            int k = 20;
            double lambda = 0.1;
            double alpha = 1.0;
            DoubleUnaryOperator confidence = x -> 1 + alpha * x;
            int numIter = 20;

            Factorization<String, String> factorization = new HKVFactorizer<String, String>(lambda, confidence, numIter).factorize(k, training);

            return new MFRecommender<>(userIndex, itemIndex, factorization);
        });
        recMap.put("hkv_20_0.01_1.0_20", () -> {
            int k = 20;
            double lambda = 0.01;
            double alpha = 1.0;
            DoubleUnaryOperator confidence = x -> 1 + alpha * x;
            int numIter = 20;

            Factorization<String, String> factorization = new HKVFactorizer<String, String>(lambda, confidence, numIter).factorize(k, training);

            return new MFRecommender<>(userIndex, itemIndex, factorization);
        });
        recMap.put("hkv_20_0.1_1.0_100", () -> {
            int k = 20;
            double lambda = 0.1;
            double alpha = 1.0;
            DoubleUnaryOperator confidence = x -> 1 + alpha * x;
            int numIter = 100;

            Factorization<String, String> factorization = new HKVFactorizer<String, String>(lambda, confidence, numIter).factorize(k, training);

            return new MFRecommender<>(userIndex, itemIndex, factorization);
        });
        recMap.put("hkv_100_0.1_1.0_20", () -> {
            int k = 100;
            double lambda = 0.1;
            double alpha = 1.0;
            DoubleUnaryOperator confidence = x -> 1 + alpha * x;
            int numIter = 20;

            Factorization<String, String> factorization = new HKVFactorizer<String, String>(lambda, confidence, numIter).factorize(k, training);

            return new MFRecommender<>(userIndex, itemIndex, factorization);
        });
        recMap.put("hkv_100_0.01_1.0_20", () -> {
            int k = 100;
            double lambda = 0.01;
            double alpha = 1.0;
            DoubleUnaryOperator confidence = x -> 1 + alpha * x;
            int numIter = 20;

            Factorization<String, String> factorization = new HKVFactorizer<String, String>(lambda, confidence, numIter).factorize(k, training);

            return new MFRecommender<>(userIndex, itemIndex, factorization);
        });
        recMap.put("hkv_100_0.1_1.0_100", () -> {
            int k = 100;
            double lambda = 0.1;
            double alpha = 1.0;
            DoubleUnaryOperator confidence = x -> 1 + alpha * x;
            int numIter = 100;

            Factorization<String, String> factorization = new HKVFactorizer<String, String>(lambda, confidence, numIter).factorize(k, training);

            return new MFRecommender<>(userIndex, itemIndex, factorization);
        });
        return recMap;
    }

    private static SimplePoynterDataset loadData(String folder) {
        SimplePoynterDataset data = new SimplePoynterDataset(folder + "userid_tweetid_claimid.csv",
                folder + "users_num_tweets_num_claims.csv");
        data = new SimplePoynterDataset(folder + "user_id__tweetid__parentid__claimid.csv",
                folder + "users_num_tweets_num_claims.csv");
        data = new SimplePoynterDataset(folder + "user_id__tweetid__parentid__claimid.csv",
                folder + "users_num_tweets_num_claims.csv");
        return data;
    }

    public static void main(String[] args) throws Exception {
        int step = -1;

        try {
            step = Integer.parseInt(args[0]);
        } catch (Exception e) {
        }

        switch (step) {
            case 10: {
                // process original dataset and generate recsys interaction files
                SimplePoynterDataset data = loadData("generated_dataset/");
                PoynterProcessor proc = new PoynterProcessor(data);

                proc.generateUserItemFile("merged__user_item__no_filter.dat", Double.NaN);
                proc.generateUserItemFile("merged__user_item__0.2m.dat", 0.2);
                proc.generateUserItemFile("merged__user_item__0.5m.dat", 0.5);
                proc.generateUserItemFile("merged__user_item__0.8m.dat", 0.8);
            }
            break;

            case 20: {
                // generate recommendations
                System.out.println("20 trainfile outfolder");
                System.out.println(Arrays.toString(args));

                String trainFile = "";
                trainFile = args[1];
                String outFolder = args[2];

                PreferenceData<String, String> trainData = SimplePreferenceData.load(SimpleRatingPreferencesReader.get().read(trainFile, sp, sp));

                List<String> usersList = trainData.getAllUsers().collect(Collectors.toList());
                List<String> itemsList = trainData.getAllItems().collect(Collectors.toList());

                FastUserIndex<String> userIndexTrain = SimpleFastUserIndex.load(usersList.stream());
                FastItemIndex<String> itemIndexTrain = SimpleFastItemIndex.load(itemsList.stream());

                FastPreferenceData<String, String> training = SimpleFastPreferenceData.load(SimpleRatingPreferencesReader.get().read(trainFile, sp, sp), userIndexTrain, itemIndexTrain);

                //
                Map<String, Supplier<Recommender<String, String>>> recMap = getRecommenders(userIndexTrain, itemIndexTrain, training);

                boolean test = false;

                if (test) {
                    Recommender<String, String> rec = new PopularityRecommender<>(training);
                    if (false) {
                        String userId = usersList.get(0);

                        Recommendation<String, String> r = rec.getRecommendation(userId);
                        for (Tuple2od<String> t : r.getItems()) {
                            System.out.println(t.v1() + " " + t.v2());
                        }
                    } else {
                        Set<String> targetUsers = new HashSet<>(usersList.subList(0, 3));
                        RecommendationFormat<String, String> format = new SimpleRecommendationFormat<>(sp, sp);
                        Function<String, IntPredicate> filter = FastFilters.all(); //notInTrain(training);
                        int maxLength = 10;
                        RecommenderRunner<String, String> runner = new FastFilterRecommenderRunner<>(userIndexTrain, itemIndexTrain, targetUsers.stream(), filter, maxLength);

                        try (RecommendationFormat.Writer<String, String> writer = format.getWriter(System.out)) {
                            runner.run(rec, writer);
                        }
                    }
                } else {
                    Set<String> targetUsers = new HashSet<>(usersList);
                    RecommendationFormat<String, String> format = new SimpleRecommendationFormat<>(sp, sp);
                    Function<String, IntPredicate> filter = FastFilters.notInTrain(training);
                    int maxLength = 50;
                    RecommenderRunner<String, String> runner = new FastFilterRecommenderRunner<>(userIndexTrain, itemIndexTrain, targetUsers.stream(), filter, maxLength);

                    final String prefix = outFolder + (outFolder.endsWith("/") ? "" : "/") + "rec__" + trainFile + "__";

                    recMap.forEach(Unchecked.biConsumer((name, recommender) -> {
                        if (new File(prefix + name).exists()) {
                            System.out.println("Skipping " + name);
                        } else {
                            System.out.println("Running " + name);
                            try (RecommendationFormat.Writer<String, String> writer = format.getWriter(prefix + name)) {
                                runner.run(recommender.get(), writer);
                            }
                        }
                    }));
                }
            }
            break;

            case 30: {
                // evaluate all users
                System.out.println("30 dataFolder trainingFile recFolder evalFolder");
                System.out.println(Arrays.toString(args));

                String dataFolder = args[1];
                String trainingFile = args[2];
                String recFolder = args[3];
                String evalFolder = args[4];

                SimplePoynterDataset data = loadData(dataFolder);
                PoynterProcessor proc = new PoynterProcessor(data);
                Set<String> groundtruth = proc.getMisinformativeItems();

                PreferenceData<String, String> trainData = SimplePreferenceData.load(SimpleRatingPreferencesReader.get().read(trainingFile, sp, sp));
                for (File recFile : new File(recFolder).listFiles()) {
                    String outFile = evalFolder + "eval__" + recFile.getName();
                    if (!recFile.getName().startsWith("rec__" + new File(trainingFile).getName() + "__")) {
                        continue;
                    }
                    if (new File(outFile).exists()) {
                        System.out.println("Skipping " + outFile);
                    }
                    Map<String, SystemMetric<String, String>> sysMetrics = new HashMap<>();

                    sysMetrics.put("misgini@5", new MisinformationGini<>(5, groundtruth, "nogroundtruth"));
                    sysMetrics.put("misgini@10", new MisinformationGini<>(10, groundtruth, "nogroundtruth"));
                    sysMetrics.put("misgini@20", new MisinformationGini<>(20, groundtruth, "nogroundtruth"));

                    sysMetrics.put("miscount@5", new AverageRecommendationMetric<>(new MisinformationCount<>(groundtruth, 5), true));
                    sysMetrics.put("miscount@10", new AverageRecommendationMetric<>(new MisinformationCount<>(groundtruth, 10), true));
                    sysMetrics.put("miscount@20", new AverageRecommendationMetric<>(new MisinformationCount<>(groundtruth, 20), true));

                    sysMetrics.put("misratiodiff@5", new AverageRecommendationMetric<>(new MisinformationRatioDelta<>(trainData, groundtruth, 5, true), true));
                    sysMetrics.put("misratiodiff@10", new AverageRecommendationMetric<>(new MisinformationRatioDelta<>(trainData, groundtruth, 10, true), true));
                    sysMetrics.put("misratiodiff@20", new AverageRecommendationMetric<>(new MisinformationRatioDelta<>(trainData, groundtruth, 20, true), true));

                    RecommendationFormat<String, String> format = new SimpleRecommendationFormat<>(sp, sp);
                    format.getReader(recFile).readAll().forEach(rec -> sysMetrics.values().forEach(metric -> metric.add(rec)));

                    PrintStream out = new PrintStream(new File(outFile));
                    sysMetrics.forEach((name, metric) -> out.println(name + "\t" + metric.evaluate()));
                    out.close();
                }
            }
            break;

            case 31: {
                // evaluate only users with misinformative items (groundtruth)
                System.out.println("31 dataFolder trainingFile recFolder evalFolder");
                System.out.println(Arrays.toString(args));

                String dataFolder = args[1];
                String trainingFile = args[2];
                String recFolder = args[3];
                String evalFolder = args[4];

                SimplePoynterDataset data = loadData(dataFolder);
                PoynterProcessor proc = new PoynterProcessor(data);
                final Set<String> groundtruth = proc.getMisinformativeItems();
                final Set<String> usersWithGroundtruth = proc.getUsersWithMisinformativeItems();

                PreferenceData<String, String> trainData = SimplePreferenceData.load(SimpleRatingPreferencesReader.get().read(trainingFile, sp, sp));
                for (File recFile : new File(recFolder).listFiles()) {
                    String outFile = evalFolder + "eval__userswith__" + recFile.getName();
                    if (!recFile.getName().startsWith("rec__" + new File(trainingFile).getName() + "__")) {
                        continue;
                    }
                    if (new File(outFile).exists()) {
                        System.out.println("Skipping " + outFile);
                    }
                    Map<String, SystemMetric<String, String>> sysMetrics = new HashMap<>();

                    sysMetrics.put("misgini@5", new MisinformationGini<>(5, groundtruth, "nogroundtruth"));
                    sysMetrics.put("misgini@10", new MisinformationGini<>(10, groundtruth, "nogroundtruth"));
                    sysMetrics.put("misgini@20", new MisinformationGini<>(20, groundtruth, "nogroundtruth"));

                    sysMetrics.put("miscount@5", new AverageRecommendationMetric<>(new MisinformationCount<>(groundtruth, 5), true));
                    sysMetrics.put("miscount@10", new AverageRecommendationMetric<>(new MisinformationCount<>(groundtruth, 10), true));
                    sysMetrics.put("miscount@20", new AverageRecommendationMetric<>(new MisinformationCount<>(groundtruth, 20), true));

                    sysMetrics.put("misratiodiff@5", new AverageRecommendationMetric<>(new MisinformationRatioDelta<>(trainData, groundtruth, 5, true), true));
                    sysMetrics.put("misratiodiff@10", new AverageRecommendationMetric<>(new MisinformationRatioDelta<>(trainData, groundtruth, 10, true), true));
                    sysMetrics.put("misratiodiff@20", new AverageRecommendationMetric<>(new MisinformationRatioDelta<>(trainData, groundtruth, 20, true), true));

                    RecommendationFormat<String, String> format = new SimpleRecommendationFormat<>(sp, sp);
                    format.getReader(recFile).readAll().forEach(rec -> {
                        if (usersWithGroundtruth.contains(rec.getUser())) {
                            sysMetrics.values().forEach(metric -> metric.add(rec));
                        }
                    });

                    PrintStream out = new PrintStream(new File(outFile));
                    sysMetrics.forEach((name, metric) -> out.println(name + "\t" + metric.evaluate()));
                    out.close();
                }
            }
            break;

            case 40: {
                // generate new dataset simulating one feedback loop
                System.out.println("40 trainingfile rec perc_accepted_recs cutoff new_trainingfile");
                System.out.println(Arrays.toString(args));

                String inputTraining = args[1];
                String outputTraining = args[5];
                String inputRec = args[2];
                double acceptedRecsRatio = Double.parseDouble(args[3]);
                int acceptedCutoff = Integer.parseInt(args[4]);

                final Random rnd = new Random(20201001L);

                PrintStream out = new PrintStream(new File(outputTraining));
                // insert the original training
                Files.readAllLines(Paths.get(inputTraining)).forEach(l -> out.println(l));
                // process the recommender
                RecommendationFormat<String, String> format = new SimpleRecommendationFormat<>(sp, sp);
                format.getReader(inputRec).readAll().forEach(rec -> rec.getItems().subList(0, acceptedCutoff).forEach(r -> {
                    if (rnd.nextDouble() < acceptedRecsRatio) {
                        out.println(rec.getUser() + "\t" + r.v1() + "\t1");
                    }
                }
                ));
                //
                out.close();
            }
            break;

            default:
                throw new AssertionError();
        }
    }
}
