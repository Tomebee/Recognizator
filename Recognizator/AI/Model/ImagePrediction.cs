using System.Linq;
using System.Text.Json.Serialization;

namespace Recognizator.AI.Model
{
    public class ImagePrediction
    {
        [JsonIgnore]
        public float[] Score { get; set; }

        public string PredictedLabel { get; set; }

        public float MaxScore => Score.Max();
    }
}
