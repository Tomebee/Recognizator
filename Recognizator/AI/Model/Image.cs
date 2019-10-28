using Microsoft.ML.Data;

namespace Recognizator.AI.Model
{
    public class Image
    {
        [LoadColumn(0)]
        public string Path;

        [LoadColumn(1)]
        public string Label;
    }
}
