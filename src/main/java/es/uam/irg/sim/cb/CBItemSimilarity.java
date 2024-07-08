package es.uam.irg.sim.cb;

import es.uam.irg.sim.cb.lucene.LuceneFeatureDataTF;
import es.uam.eps.ir.ranksys.core.feature.FeatureData;
import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.nn.item.sim.ItemSimilarity;
import es.uam.eps.ir.ranksys.nn.sim.Similarity;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.longs.Long2DoubleOpenHashMap;
import static java.lang.Math.sqrt;
import java.util.function.IntToDoubleFunction;
import java.util.stream.Stream;
import org.jooq.lambda.tuple.Tuple2;
import org.ranksys.core.util.tuples.Tuple2id;
import static org.ranksys.core.util.tuples.Tuples.tuple;

/**
 *
 */
public class CBItemSimilarity {

    private final ItemSimilarity<Long> itemSim;

    // data should be received as user-item
    public CBItemSimilarity(final FastPreferenceData<Long, Long> data, final String indexPath) {
        FeatureData<Long, Long, Double> content = new LuceneFeatureDataTF(indexPath, false);
        // this is not a generic similarity, but an item similarity
        Similarity sim = new ContentSimilarity(data, content);
        this.itemSim = new ItemSimilarity<>(data, sim);
    }

    public ItemSimilarity<Long> getItemSim() {
        return itemSim;
    }

    public static class ContentSimilarity implements Similarity {

        /**
         * User-item preferences.
         */
        protected final FastPreferenceData<Long, Long> data;
        protected final Int2DoubleMap norm2Map;
        protected final FeatureData<Long, Long, Double> content;

        public ContentSimilarity(final FastPreferenceData<Long, Long> data, final FeatureData<Long, Long, Double> content) {
            this.data = data;
            this.content = content;

            this.norm2Map = new Int2DoubleOpenHashMap();
            norm2Map.defaultReturnValue(0.0);
            data.getIidxWithPreferences().forEach(idx -> norm2Map.put(idx, getNorm2(idx)));
        }

        private double getNorm2(int iidx) {
            return content.getItemFeatures(data.iidx2item(iidx))
                    .mapToDouble(Tuple2::v2)
                    .map(x -> x * x)
                    .sum();
        }

        @Override
        public IntToDoubleFunction similarity(int idx) {
            Long2DoubleOpenHashMap map = new Long2DoubleOpenHashMap();
            content.getItemFeatures(data.iidx2item(idx)).forEach(iv -> map.put(iv.v1, iv.v2));

            double norm2A = norm2Map.get(idx);

            return idx2 -> {
                double product = content.getItemFeatures(data.iidx2item(idx2)).
                        mapToDouble(iv -> iv.v2 * map.get(iv.v1)).sum();

                return sim(product, norm2A, norm2Map.get(idx2));
            };
        }

        private static double sim(double product, double norm2A, double norm2B) {
            return cosineSim(product, norm2A, norm2B);
        }

        private static double cosineSim(double product, double norm2A, double norm2B) {
            return product / sqrt(norm2A * norm2B);
        }

        @Override
        public Stream<Tuple2id> similarElems(int idx) {
            double norm2A = norm2Map.get(idx);

            return getProductMap(idx).int2DoubleEntrySet().stream()
                    .map(e -> {
                        int idx2 = e.getIntKey();
                        double product = e.getDoubleValue();
                        double norm2B = norm2Map.get(idx2);
                        return tuple(idx2, sim(product, norm2A, norm2B));
                    });
        }

        private Int2DoubleMap getProductMap(int iidx) {
            Int2DoubleOpenHashMap productMap = new Int2DoubleOpenHashMap();
            productMap.defaultReturnValue(0.0);

            content.getItemFeatures(data.iidx2item(iidx))
                    .forEach(ip -> content.getItemFeatures(ip.v1)
                    .forEach(up -> productMap.addTo(data.item2iidx(up.v1), ip.v2 * up.v2)));

            productMap.remove(iidx);

            return productMap;
        }

    }

}
