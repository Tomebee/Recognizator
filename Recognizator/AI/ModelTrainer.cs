using Microsoft.ML;
using Recognizator.AI.Model;
using System;
using System.IO;
using System.Linq;

namespace Recognizator.AI
{
    public interface IModelTrainer
    {
        ITransformer Train(MLContext context);
    }

    internal class ModelTrainer : IModelTrainer
    {
        private readonly IPipelineBuilder _pipelineBuilder;
        private readonly string _imagesPath = Path.Combine(Environment.CurrentDirectory, "..", "Images");

        public ModelTrainer(IPipelineBuilder pipelineBuilder)
        {
            _pipelineBuilder = pipelineBuilder;
        }

        public ITransformer Train(MLContext context)
        {
            var images = Directory.GetFiles(_imagesPath, "*", SearchOption.AllDirectories)
                .Select(file => new Image
                {
                    Path = file,
                    Label = Directory.GetParent(file).Name
                });

            var imageData = context.Data.LoadFromEnumerable(images);
            var imageDataShuffled = context.Data.ShuffleRows(imageData);

            var pipeline = _pipelineBuilder
                .UseImagesFolder(_imagesPath)
                .Build(context);

            return pipeline.Fit(imageDataShuffled);
        }
    }
}

//var imageData = context.Data.LoadFromEnumerable(images);
//var imageDataShuffled = context.Data.ShuffleRows(imageData);
//var a = context.Transforms.Conversion
//    .MapValueToKey(nameof(Image.Label),
//        nameof(Image.Label),
//        keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
//    .Append(context.Transforms.LoadImages(
//        "Image",
//        _imagesPath,
//        nameof(Image.Path)))
//    .Append(context.Transforms.ResizeImages("Image", imageWidth: 224, imageHeight: 224))
//    .Append(context.Transforms.ExtractPixels("Pixels", "Image"))
//    .Append(context.Transforms.DnnFeaturizeImage("ImageData", dnnInput =>
//    {
//        return dnnInput.ModelSelector.ResNet18(context, dnnInput.OutputColumn, dnnInput.InputColumn);
//    }, "Pixels"))
//    .Append(context.Model.ImageClassification(
//        "ImageData",
//        nameof(Image.Label),
//        arch: DnnEstimator.Architecture.ResnetV2101,
//        epoch: 100
//        ));
//var testTrainData = context.Data.TrainTestSplit(imageDataShuffled, testFraction: 0.2);
//var conversion = context.Transforms.Conversion.MapValueToKey(nameof(Image.Label), nameof(Image.Label), keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue);