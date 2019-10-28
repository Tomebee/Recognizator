using Microsoft.ML;
using Recognizator.AI.Model;
using System;

namespace Recognizator.AI
{
    public interface IImagePredictor
    {
        ImagePrediction Predict(string pathToImage);
    }

    internal class ImagePredictor : IImagePredictor
    {
        private readonly MLContext _context;
        private readonly IPredictionEngineBuilder _predictionEngineBuilder;
        private PredictionEngine<Image, ImagePrediction> _predictionEngine;

        private PredictionEngine<Image, ImagePrediction> PredictionEngine
        {
            get
            {
                if (_predictionEngine is null)
                    _predictionEngine = _predictionEngineBuilder.Build(_context);
                return _predictionEngine;
            }
        }
        
        public ImagePredictor(IPredictionEngineBuilder predictionEngineBuilder)
        {
            _predictionEngineBuilder = predictionEngineBuilder;
            _context = new MLContext();
        }

        public ImagePrediction Predict(string pathToImage)
        {
            if (string.IsNullOrEmpty(pathToImage))
            {
                throw new InvalidOperationException("Invalid path.");
            }
            
            var prediction = PredictionEngine.Predict(new Image
            {
                Path = pathToImage
            });

            return prediction;
        }
    }
}
