package es.uam.irg.sim.cb;

import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.nn.item.ItemNeighborhoodRecommender;
import es.uam.eps.ir.ranksys.nn.item.neighborhood.ItemNeighborhood;
import es.uam.eps.ir.ranksys.nn.item.neighborhood.ItemNeighborhoods;
import es.uam.eps.ir.ranksys.nn.item.sim.ItemSimilarity;
import es.uam.eps.ir.ranksys.rec.Recommender;

/**
 *
 */
public class CBItemBasedRecommender {

    private Recommender<Long, Long> recommender;

    public CBItemBasedRecommender(final FastPreferenceData<Long, Long> data, final String indexPath, int k) {
        int q = 1;

        ItemSimilarity<Long> sim = new CBItemSimilarity(data, indexPath).getItemSim();
        ItemNeighborhood<Long> neighborhood = ItemNeighborhoods.cached(ItemNeighborhoods.topK(sim, k));

        recommender = new ItemNeighborhoodRecommender<>(data, neighborhood, q);
    }

    public Recommender<Long, Long> getRecommender() {
        return recommender;
    }

}
