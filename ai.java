import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.eval.DataModelBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class RecommendationSystem {
    public static void main(String[] args) {
        try {
            // 1. Load data model from file
            DataModel model = new FileDataModel(new File("data/dataset.csv"));

            // 2. Compute user similarity
            UserSimilarity similarity = new PearsonCorrelationSimilarity(model);

            // 3. Define neighborhood
            UserNeighborhood neighborhood = new NearestNUserNeighborhood(2, similarity, model);

            // 4. Build recommender
            Recommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);

            // 5. Recommend items for all users
            for (LongPrimitiveIterator users = model.getUserIDs(); users.hasNext();) {
                long userId = users.nextLong();
                List<RecommendedItem> recommendations = recommender.recommend(userId, 3);
                System.out.println("User ID: " + userId);
                for (RecommendedItem recommendation : recommendations) {
                    System.out.println("  Recommended Item: " + recommendation.getItemID() +
                            " (Value: " + recommendation.getValue() + ")");
                }
            }

            // 6. Evaluate recommender (optional)
            RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
            double score = evaluator.evaluate(new RecommenderBuilder() {
                @Override
                public Recommender buildRecommender(DataModel dataModel) throws TasteException {
                    UserSimilarity similarity = new PearsonCorrelationSimilarity(dataModel);
                    UserNeighborhood neighborhood = new NearestNUserNeighborhood(2, similarity, dataModel);
                    return new GenericUserBasedRecommender(dataModel, neighborhood, similarity);
                }
            }, null, model, 0.7, 1.0);

            System.out.println("Evaluation Score: " + score);

        } catch (IOException | TasteException e) {
            e.printStackTrace();
        }
    }
}
