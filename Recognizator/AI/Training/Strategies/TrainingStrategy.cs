using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Recognizator.AI.Model;

namespace Recognizator.AI.Training.Strategies
{
    public interface ITrainingStrategy
    {
        ITransformer Train(MLContext context, string imagesFolder);
    }

    internal abstract class TrainingStrategy : ITrainingStrategy
    {
        public abstract ITransformer Train(MLContext context, string imagesFolder);

        public virtual IEnumerable<Image> PrepareImages(string imagesFolder) => Directory
            .GetFiles(imagesFolder, "*", SearchOption.AllDirectories)
            .Select(file => new Image
            {
                Path = file,
                Label = Directory.GetParent(file).Name
            });
    }
}
