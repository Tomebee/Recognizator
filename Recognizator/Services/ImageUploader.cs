using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace Recognizator.Services
{
    public interface IImageUploader
    {
        Task<string> Upload(IFormFile imageFile);
    }
    internal sealed class ImageUploader : IImageUploader
    {
        private readonly string[] _availableExtensions = {
            ".jpg",
            ".png",
            ".jpeg"
        };
        private readonly IWebHostEnvironment _hostEnvironment;

        public ImageUploader(IWebHostEnvironment hostEnvironment) => _hostEnvironment = hostEnvironment;

        public async Task<string> Upload(IFormFile imageFile)
        {
            if(imageFile is null)
            {
                throw new ArgumentNullException(nameof(imageFile));
            }

            var extension = Path.GetExtension(imageFile.FileName);
            if(!_availableExtensions.Contains(extension))
            {
                throw new InvalidOperationException($"Cannot upload {extension} file. Available extensions: {string.Join(',', _availableExtensions)} ");
            }

            var path = Path.Combine(_hostEnvironment.WebRootPath, "images", $"{Guid.NewGuid() + extension}");
            await using (var stream = new FileStream(path, FileMode.Create))
                await imageFile.CopyToAsync(stream);

            return path;
        }
    }
}
