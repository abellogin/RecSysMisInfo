package es.uam.irg.misinf.eval;

import es.uam.eps.ir.ranksys.core.Recommendation;
import es.uam.eps.ir.ranksys.metrics.AbstractRecommendationMetric;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import org.ranksys.core.util.tuples.Tuple2od;

/**
 *
 * @author Alejandro
 */
public class MisinformationCount<U, I> extends AbstractRecommendationMetric<U, I> {

    private final Set<I> groundtruth;
    private final int cutoff;

    public MisinformationCount(Set<I> groundtruth, int cutoff) {
        this.groundtruth = groundtruth;
        this.cutoff = cutoff;
    }

    @Override
    public double evaluate(Recommendation<U, I> recommendation) {
        if (recommendation.getItems().isEmpty()) {
            return 0;
        }

        List<I> lstRec = recommendation.getItems().stream().limit(cutoff).map(Tuple2od::v1).collect(Collectors.toList());

        double result = 0.0;
        for (I item : lstRec) {
            if (groundtruth.contains(item)) {
                result += 1;
            }
        }
        return result / cutoff;
    }

}
