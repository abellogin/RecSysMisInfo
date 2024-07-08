package es.uam.irg.misinf.eval;

import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.metrics.SystemMetric;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.stream.IntStream;
import org.ranksys.core.util.tuples.Tuple2od;

/**
 *
 * @author Alejandro
 */
public class MisinformationGini<U, I> implements SystemMetric<U, I> {

    protected final int cutoff;
    private final Set<I> groundtruth;
    protected final Object2DoubleOpenHashMap<I> itemCount;
    private final I noGroundtruthItem;
    private double freeNorm;

    public MisinformationGini(int cutoff, Set<I> groundtruth, I noGroundtruthItem) {
        this.cutoff = cutoff;
        this.groundtruth = groundtruth;
        this.noGroundtruthItem = noGroundtruthItem;

        itemCount = new Object2DoubleOpenHashMap<>();
        itemCount.defaultReturnValue(0.0);
        for (I i : groundtruth) {
            itemCount.put(i, 0.0);
        }
        itemCount.put(noGroundtruthItem, 0.0);

        freeNorm = 0.0;
    }

    @Override
    public void add(Recommendation<U, I> recommendation) {
        List<Tuple2od<I>> list = recommendation.getItems();
        int rank = Math.min(cutoff, list.size());

        IntStream.range(0, rank).forEach(k -> {
            I i = list.get(k).v1;
            if (groundtruth.contains(i)) {
                itemCount.addTo(i, 1.0);
            } else {
                itemCount.addTo(noGroundtruthItem, 1.0);
            }
        });

        freeNorm += rank;
    }

    @Override
    public double evaluate() {
        // compute Gini as in 
        // https://github.com/RankSys/RankSys/blob/master/RankSys-diversity/src/main/java/es/uam/eps/ir/ranksys/diversity/sales/metrics/GiniIndex.java
        double gi = 0;
        double[] cs = itemCount.values().toDoubleArray();
        Arrays.sort(cs);
        int numItems = groundtruth.size() + 1;
        for (int j = 0; j < cs.length; j++) {
            gi += (2 * (j + (numItems - cs.length) + 1) - numItems - 1) * (cs[j] / freeNorm);
        }
        gi /= (numItems - 1);
        gi = 1 - gi;

        return gi;
    }

    @Override
    public void combine(SystemMetric<U, I> other) {
    }

    @Override
    public void reset() {
        itemCount.clear();
        freeNorm = 0;
    }
}
