using Microsoft.ML;
using System;
using System.IO;
using Microsoft.ML.Vision;
using Recognizator.AI.Training.Strategies;
using Recognizator.AI.Training.Strategies.Inception;
using Microsoft.Extensions.Logging;

namespace Recognizator.AI.Training
{
    public interface IModelTrainer
    {
        ITransformer Train(MLContext context, TrainingMethod method);
    }

    internal sealed class ModelTrainer : IModelTrainer
    {
        private readonly string _imagesPath
#if DEBUG
        = Path.Combine(Environment.CurrentDirectory, "..", "Images");
#else
        = Path.Combine(Environment.CurrentDirectory, "Images");
#endif
        private readonly ILogger<ModelTrainer> _logger;
        
        public ModelTrainer(ILogger<ModelTrainer> logger) => _logger = logger;

        public ITransformer Train(MLContext context, TrainingMethod method) =>
            method switch
            {
                TrainingMethod.GpuSupportedInceptionRetrain => new GpuSupportedModelRetrainStrategy(ImageClassificationTrainer.Architecture.InceptionV3, _logger).Train(context, _imagesPath),
                TrainingMethod.GpuSupportedResnetRetrain => new GpuSupportedModelRetrainStrategy(ImageClassificationTrainer.Architecture.ResnetV250, _logger).Train(context, _imagesPath),
                TrainingMethod.TransferLearningWithInception => new TransferLearningWithInceptionStrategy(_logger).Train(context, _imagesPath),
                _ => throw new Exception($"Training strategy for {method.ToString()} method not implemented")
            };
    }
}