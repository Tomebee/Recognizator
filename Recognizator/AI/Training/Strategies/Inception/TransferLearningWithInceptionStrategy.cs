using System;
using System.Diagnostics;
using System.IO;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using Recognizator.AI.Model;

namespace Recognizator.AI.Training.Strategies.Inception
{
    internal sealed class TransferLearningWithInceptionStrategy : TrainingStrategy
    {
        private struct InceptionSettings
        {
            public const int ImageHeight = 224;
            public const int ImageWidth = 224;
            public const float Mean = 117;
            public const bool ChannelsLast = true;
        }
        private readonly ILogger _logger;
        
        public TransferLearningWithInceptionStrategy(ILogger logger) => _logger = logger;

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

        private IEstimator<ITransformer> BuildPipeline(MLContext context, string imagesFolder) => context.Transforms.LoadImages(
                "input",
                imagesFolder,
                nameof(Image.Path))
            .Append(context.Transforms.ResizeImages(
                "input",
                InceptionSettings.ImageWidth,
                InceptionSettings.ImageHeight,
                "input"))
            .Append(context.Transforms.ExtractPixels(
                "input",
                interleavePixelColors: InceptionSettings.ChannelsLast,
                offsetImage: InceptionSettings.Mean))
            .Append(context.Model.LoadTensorFlowModel(Path.Combine(Environment.CurrentDirectory, "AI", "Training", "Strategies", "Inception", "Model", "tensorflow_inception_graph.pb"))
                .ScoreTensorFlowModel(new[] { "softmax2_pre_activation" }, new[] { "input" }, true))
            .Append(context.Transforms.Conversion.MapValueToKey("LabelKey", "Label"))
            .Append(context.MulticlassClassification.Trainers.LbfgsMaximumEntropy(
                "LabelKey", 
                "softmax2_pre_activation"))
            .Append(context.Transforms.Conversion.MapKeyToValue(
                nameof(ImagePrediction.PredictedLabel), 
                nameof(ImagePrediction.PredictedLabel)))
            .AppendCacheCheckpoint(context);
    }
}
