package es.uam.irg.misinf.eval;

import com.google.common.util.concurrent.AtomicDouble;
import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.core.preference.PreferenceData;
import es.uam.eps.ir.ranksys.metrics.AbstractRecommendationMetric;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import org.ranksys.core.util.tuples.Tuple2od;

/**
 *
 * @author Alejandro
 */
public class MisinformationRatioDelta<U, I> extends AbstractRecommendationMetric<U, I> {

    private final Map<U, Double> trainingRatio;
    private final Set<I> groundtruth;
    private final int cutoff;
    private final boolean computeDeltaAsDiff;

    public MisinformationRatioDelta(PreferenceData<U, I> trainingProfiles, Set<I> groundtruth, int cutoff, boolean computeDeltaAsDiff) {
        this.groundtruth = groundtruth;
        this.cutoff = cutoff;
        this.computeDeltaAsDiff = computeDeltaAsDiff;
        this.trainingRatio = new HashMap<>();
        computeTrainingRatio(trainingProfiles);
    }

    private void computeTrainingRatio(PreferenceData<U, I> training) {
        training.getAllUsers().forEach(u -> {
            AtomicDouble r = new AtomicDouble(0.0);
            AtomicInteger n = new AtomicInteger(0);
            training.getUserPreferences(u).forEach(p -> {
                if (groundtruth.contains(p.v1)) {
                    r.addAndGet(1.0);
                }
                n.getAndIncrement();
            });
            trainingRatio.put(u, r.doubleValue() / n.get());
        });
    }

    @Override
    public double evaluate(Recommendation<U, I> recommendation) {
        if (recommendation.getItems().isEmpty()) {
            return 0;
        }
        if (!trainingRatio.containsKey(recommendation.getUser())) {
            return 0;
        }

        double tr = trainingRatio.get(recommendation.getUser());

        List<I> lstRec = recommendation.getItems().stream().limit(cutoff).map(Tuple2od::v1).collect(Collectors.toList());

        double ratio = 0.0;
        int n = 0;
        for (I item : lstRec) {
            if (groundtruth.contains(item)) {
                ratio += 1;
            }
            n++;
        }
        ratio /= n;

        double delta = tr / ratio;
        if (computeDeltaAsDiff) {
            delta = tr - ratio;
        }

        return delta;
    }
}
