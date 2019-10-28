using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Recognizator.AI;
using Recognizator.Services;
using System.Threading.Tasks;

namespace Recognizator.Controllers
{

    [Route("api/v1/recognize")]
    [ApiController]
    public class ImageRecognitionController : ControllerBase
    {
        private readonly IImageUploader _imageUploader;
        private readonly IImagePredictor _imagePredictor;

        public ImageRecognitionController(IImageUploader imageUploader,
            IImagePredictor imagePredictor)
        {
            _imageUploader = imageUploader;
            _imagePredictor = imagePredictor;
        }


        [HttpPost]
        public async Task<IActionResult> Recognize([FromForm] IFormFile image)
        {
            var filePath = await _imageUploader.Upload(image);

            var prediction = _imagePredictor.Predict(filePath);

            return Ok(prediction);
        }

    }
}
