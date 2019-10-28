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

    internal sealed class ModelTrainer : IModelTrainer
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