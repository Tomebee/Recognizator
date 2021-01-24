using Microsoft.ML;
using Recognizator.AI.Model;
using Recognizator.AI.Training;

namespace Recognizator.AI
{
    public interface IPredictionEngineBuilder
    {
        PredictionEngine<Image, ImagePrediction> Build(MLContext context, TrainingMethod method);
    }

    internal sealed class PredictionEngineBuilder : IPredictionEngineBuilder
    {
        private readonly IModelTrainer _modelTrainer;

        public PredictionEngineBuilder(IModelTrainer modelTrainer) => _modelTrainer = modelTrainer;

        public PredictionEngine<Image, ImagePrediction> Build(MLContext context, TrainingMethod method) 
            => context.Model.CreatePredictionEngine<Image, ImagePrediction>(_modelTrainer.Train(context, method));
    }
}
