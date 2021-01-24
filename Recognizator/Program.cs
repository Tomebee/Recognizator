using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace Recognizator
{
    public class Program
    {
        public static void Main(string[] args)
        {
            CreateHostBuilder(args).Build().Run();
        }

        public static IHostBuilder CreateHostBuilder(string[] args) =>
            Host.CreateDefaultBuilder(args)
                .ConfigureLogging(logging => {
                    logging.ClearProviders();
                    logging.AddFilter(loglevel => loglevel == LogLevel.Trace);
                    logging.AddConsole();
                })
                .ConfigureWebHostDefaults(webBuilder => webBuilder.ConfigureKestrel(serverOptions => 
                { 
                    serverOptions.Limits.MaxRequestBodySize = long.MaxValue;
                    serverOptions.Limits.MaxRequestBufferSize = long.MaxValue;
                }).UseStartup<Startup>());
    }
}
