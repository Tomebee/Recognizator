using Microsoft.ML;

namespace Recognizator.AI
{
    public interface IPipelineBuilder
    {
        IPipelineBuilder UseImagesFolder(string pathToImagesFolder);
        IEstimator<ITransformer> Build(MLContext context);
    }
}
