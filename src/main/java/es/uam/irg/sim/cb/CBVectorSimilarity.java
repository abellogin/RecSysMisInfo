package es.uam.irg.sim.cb;

import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.fast.preference.IdxPref;
import it.unimi.dsi.fastutil.doubles.DoubleIterator;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.ints.IntIterator;
import org.ranksys.core.util.tuples.Tuple2id;
import es.uam.eps.ir.ranksys.nn.sim.Similarity;

import java.util.function.IntToDoubleFunction;
import java.util.stream.Stream;

import static java.util.stream.IntStream.range;
import static org.ranksys.core.util.tuples.Tuples.tuple;

/**
 *
 * Based on VectorSimilarity from RankSys
 */
public abstract class CBVectorSimilarity implements Similarity {

    /**
     * User-item preferences.
     */
    protected final FastPreferenceData<?, ?> data;
    /**
     * feature-item preferences.
     */
    protected final FastPreferenceData<?, ?> content;

    /**
     * If true, dense vectors are used to calculate similarities.
     */
    protected final boolean dense;

    /**
     * Cached normalization for when dense is false.
     */
    protected final Int2DoubleMap norm2Map;

    /**
     * Cached normalization for when dense is true.
     */
    protected final double[] norm2Array;

    /**
     * Constructor. Uses maps for internal calculation.
     *
     * @param content content data
     * @param data preference data
     * @param dense true for array-based calculations, false to map-based
     */
    public CBVectorSimilarity(FastPreferenceData<?, ?> content, FastPreferenceData<?, ?> data, boolean dense) {
        this.content = content;
        this.data = data;
        this.dense = dense;
        if (dense) {
            this.norm2Map = null;
            this.norm2Array = new double[data.numUsers()];
            data.getUidxWithPreferences().forEach(idx -> norm2Array[idx] = getNorm2(idx));
        } else {
            this.norm2Map = new Int2DoubleOpenHashMap();
            this.norm2Array = null;
            norm2Map.defaultReturnValue(0.0);
            data.getUidxWithPreferences().forEach(idx -> norm2Map.put(idx, getNorm2(idx)));
        }
    }

    @Override
    public IntToDoubleFunction similarity(int idx1) {
        Int2DoubleOpenHashMap map = new Int2DoubleOpenHashMap();
        //data.getUidxPreferences(idx1).forEach(iv -> map.put(iv.v1, iv.v2));
        // now the users in this preferencedata are the features (always!)
        content.getUidxPreferences(idx1).forEach(iv -> map.put(iv.v1, iv.v2));
        
        // TODO!!!!
        // check CBItemSimilarity, since it has almost all the functionality
        // I think to make it more general we need:
        // a boolean to know if the similarities will be on users or items
        // an aggregator for the case of user similarity (this agg would take item profiles and combine them)
        // a featuredata object as the one used in the other class
        // it can also receive a lucene similarity to obtain different weights, if sim is null, instead of a search the raw TF is used as weight

        double norm2A = norm2Map.get(idx1);

        return idx2 -> {
            double product = data.getUidxPreferences(idx2)
                    .mapToDouble(iv -> iv.v2 * map.get(iv.v1))
                    .sum();

            return sim(product, norm2A, norm2Map.get(idx2));
        };
    }

    private Int2DoubleMap getProductMap(int uidx) {
        Int2DoubleOpenHashMap productMap = new Int2DoubleOpenHashMap();
        productMap.defaultReturnValue(0.0);

        if (data.useIteratorsPreferentially()) {
            IntIterator iidxs = data.getUidxIidxs(uidx);
            DoubleIterator ivs = data.getUidxVs(uidx);
            while (iidxs.hasNext()) {
                int iidx = iidxs.nextInt();
                double iv = ivs.nextDouble();
                IntIterator vidxs = data.getIidxUidxs(iidx);
                DoubleIterator vvs = data.getIidxVs(iidx);
                while (vidxs.hasNext()) {
                    productMap.addTo(vidxs.nextInt(), iv * vvs.nextDouble());
                }
            }
        } else {
            data.getUidxPreferences(uidx)
                    .forEach(ip -> data.getIidxPreferences(ip.v1)
                    .forEach(up -> productMap.addTo(up.v1, ip.v2 * up.v2)));
        }

        productMap.remove(uidx);

        return productMap;
    }

    private double[] getProductArray(int uidx) {
        double[] productArray = new double[data.numUsers()];

        if (data.useIteratorsPreferentially()) {
            IntIterator iidxs = data.getUidxIidxs(uidx);
            DoubleIterator ivs = data.getUidxVs(uidx);
            while (iidxs.hasNext()) {
                int iidx = iidxs.nextInt();
                double iv = ivs.nextDouble();
                IntIterator vidxs = data.getIidxUidxs(iidx);
                DoubleIterator vvs = data.getIidxVs(iidx);
                while (vidxs.hasNext()) {
                    productArray[vidxs.nextInt()] += iv * vvs.nextDouble();
                }
            }
        } else {
            data.getUidxPreferences(uidx)
                    .forEach(ip -> data.getIidxPreferences(ip.v1)
                    .forEach(up -> productArray[up.v1] += ip.v2 * up.v2));
        }

        productArray[uidx] = 0.0;

        return productArray;
    }

    private double getNorm2(int uidx) {
        if (data.useIteratorsPreferentially()) {
            DoubleIterator ivs = data.getUidxVs(uidx);
            double sum = 0;
            while (ivs.hasNext()) {
                double iv = ivs.nextDouble();
                sum += iv * iv;
            }
            return sum;
        } else {
            return data.getUidxPreferences(uidx)
                    .mapToDouble(IdxPref::v2)
                    .map(x -> x * x)
                    .sum();
        }
    }

    @Override
    public Stream<Tuple2id> similarElems(int idx1) {
        if (dense) {
            double norm2A = norm2Array[idx1];

            double[] productArray = getProductArray(idx1);
            return range(0, productArray.length)
                    .filter(idx2 -> productArray[idx2] != 0.0)
                    .mapToObj(idx2 -> tuple(idx2, sim(productArray[idx2], norm2A, norm2Array[idx2])));
        } else {
            double norm2A = norm2Map.get(idx1);

            return getProductMap(idx1).int2DoubleEntrySet().stream()
                    .map(e -> {
                        int idx2 = e.getIntKey();
                        double product = e.getDoubleValue();
                        double norm2B = norm2Map.get(idx2);
                        return tuple(idx2, sim(product, norm2A, norm2B));
                    });
        }
    }

    /**
     * Calculates the similarity value.
     *
     * @param product value of the inner product of vectors
     * @param norm2A square of the norm of the first vector
     * @param norm2B square of the norm of the second vector
     * @return similarity value
     */
    protected abstract double sim(double product, double norm2A, double norm2B);
}
