using Microsoft.ML;
using Recognizator.AI.Model;
using System;
using System.Collections.Generic;

namespace Recognizator.AI
{
    public interface IImagePredictor
    {
        ImagePrediction Predict(string pathToImage, TrainingMethod method = TrainingMethod.TransferLearningWithInception);
    }

    internal sealed class ImagePredictor : IImagePredictor
    {
        private readonly MLContext _context;
        private readonly IPredictionEngineBuilder _predictionEngineBuilder;
        private readonly IDictionary<TrainingMethod, PredictionEngine<Image, ImagePrediction>> _predictionEngines;
        
        public ImagePredictor(IPredictionEngineBuilder predictionEngineBuilder)
        {
            _predictionEngineBuilder = predictionEngineBuilder;
            _context = new MLContext();
            _predictionEngines = new Dictionary<TrainingMethod, PredictionEngine<Image, ImagePrediction>>();
        }

        public ImagePrediction Predict(string pathToImage, TrainingMethod method) => GetPredictionEngine(method)
            .Predict(new Image
            {
                Path = pathToImage ?? throw new InvalidOperationException("Invalid path.")
            });

        private PredictionEngine<Image, ImagePrediction> GetPredictionEngine(TrainingMethod method) 
        {
            if(!_predictionEngines.TryGetValue(method, out var engine)) 
            {
                engine = _predictionEngineBuilder.Build(_context, method);
                _predictionEngines[method] = engine;
            }

            return engine;
        }
    }
}
