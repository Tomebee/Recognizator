using System.Diagnostics;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using Microsoft.ML.Transforms;
using Microsoft.ML.Vision;
using Recognizator.AI.Model;

namespace Recognizator.AI.Training.Strategies
{
    internal sealed class GpuSupportedModelRetrainStrategy : TrainingStrategy
    {
        private readonly ImageClassificationTrainer.Architecture _architecture;
        private readonly ILogger _logger; 
        
        public GpuSupportedModelRetrainStrategy(ImageClassificationTrainer.Architecture architecture, ILogger logger)
        {
            _architecture = architecture;
            _logger = logger;
        }

        public override ITransformer Train(MLContext context, string imagesFolder)
        {
            _logger.LogTrace("==========================================");
            _logger.LogTrace("1.          Preparing data");

            var watch = Stopwatch.StartNew();
            var images = PrepareImages(imagesFolder);
            var imageData = context.Data.LoadFromEnumerable(images);
            var imageDataShuffled = context.Data.ShuffleRows(imageData);

            watch.Stop();
            _logger.LogTrace($"      Took {watch.ElapsedMilliseconds} ms");
            _logger.LogTrace("2.  Building pipeline and training model");
            watch = Stopwatch.StartNew();

            var pipeline = BuildPipeline(context, imagesFolder);
            var trainedModel = pipeline.Fit(imageDataShuffled);

            watch.Stop();
            _logger.LogTrace($"      Took {watch.ElapsedMilliseconds} ms");

            return trainedModel;
        }

        private IEstimator<ITransformer> BuildPipeline(MLContext context, string imagesFolder) => context.Transforms
            .Conversion
            .MapValueToKey("LabelAsKey", nameof(Image.Label),
                keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
            .Append(context.Transforms.LoadRawImageBytes("Image", imagesFolder, nameof(Image.Path)))
            .Append(context.MulticlassClassification.Trainers.ImageClassification(
                new ImageClassificationTrainer.Options
                {
                    FeatureColumnName = "Image",
                    LabelColumnName = "LabelAsKey",
                    Arch = _architecture,
                    Epoch = 100,
                    LearningRate = 1,
                    BatchSize = 10,
                    PredictedLabelColumnName = nameof(ImagePrediction.PredictedLabel)
                }))
            .Append(context.Transforms.Conversion.MapKeyToValue(nameof(ImagePrediction.PredictedLabel),
                nameof(ImagePrediction.PredictedLabel)));
    }
}
