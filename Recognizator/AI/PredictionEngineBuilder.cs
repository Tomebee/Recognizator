using Microsoft.ML;
using Recognizator.AI.Model;

namespace Recognizator.AI
{
    public interface IPredictionEngineBuilder
    {
        PredictionEngine<Image, ImagePrediction> Build(MLContext context);
    }

    public class PredictionEngineBuilder : IPredictionEngineBuilder
    {
        private readonly IModelTrainer _modelTrainer;

        public PredictionEngineBuilder(IModelTrainer modelTrainer)
        {
            _modelTrainer = modelTrainer;
        }

        public PredictionEngine<Image, ImagePrediction> Build(MLContext context)
        {
            var model = _modelTrainer.Train(context);
            return context.Model.CreatePredictionEngine<Image, ImagePrediction>(model);
        }
    }
}
