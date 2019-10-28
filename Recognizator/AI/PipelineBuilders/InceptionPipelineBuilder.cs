using Microsoft.ML;
using Recognizator.AI.Model;
using System;
using System.IO;
using System.Linq;

namespace Recognizator.AI
{

    internal class InceptionPipelineBuilder : IPipelineBuilder
    {
        private struct InceptionSettings
        {
            public const int ImageHeight = 224;
            public const int ImageWidth = 224;
            public const float Mean = 117;
            public const float Scale = 1;
            public const bool ChannelsLast = true;
        }

        private string _imagesFolder;
        private string ImagesFolder
        {
            get
            {
                if (string.IsNullOrEmpty(_imagesFolder))
                    throw new Exception("No path set.");
                return _imagesFolder;
            }
            set => _imagesFolder = value;
        }

        public IEstimator<ITransformer> Build(MLContext context) => context.Transforms.LoadImages(
                    outputColumnName: "input",
                    imageFolder: ImagesFolder,
                    inputColumnName: nameof(Image.Path))
                .Append(context.Transforms.ResizeImages(
                    outputColumnName: "input",
                    imageWidth: InceptionSettings.ImageWidth,
                    imageHeight: InceptionSettings.ImageHeight,
                    inputColumnName: "input"))
                .Append(context.Transforms.ExtractPixels(
                    outputColumnName: "input",
                    interleavePixelColors: InceptionSettings.ChannelsLast,
                    offsetImage: InceptionSettings.Mean))
                .Append(context.Model.LoadTensorFlowModel(Path.Combine(Environment.CurrentDirectory, "AI", "Inception", "tensorflow_inception_graph.pb")).
                    ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true))
                .Append(context.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label"))
                .Append(context.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "LabelKey", featureColumnName: "softmax2_pre_activation"))
                .Append(context.Transforms.Conversion.MapKeyToValue(nameof(ImagePrediction.PredictedLabel), nameof(ImagePrediction.PredictedLabel)))
                .AppendCacheCheckpoint(context);


        public IPipelineBuilder UseImagesFolder(string pathToImagesFolder)
        {
            _imagesFolder = pathToImagesFolder;
            return this;
        }
    }
}