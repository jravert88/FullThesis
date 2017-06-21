using System;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Threading.Tasks;
using System.Json;

class StartJob
{
    static void Main(string[] args)
    {
        const string apiKey = "ab316207aeb72f7c32d9f6ebd64443c6044455bc";
        const string endpoint = "https://sandbox.zamzar.com/v1/jobs";
        const string sourceFile = @"/tmp/portrait.gif";
        const string targetFormat = "png";

        JsonValue json = Upload(apiKey, endpoint, sourceFile, targetFormat).Result;
        Console.WriteLine(json);
    }

    static async Task<JsonValue> Upload(string key, string url, string sourceFile, string targetFormat)
    {
        using (HttpClientHandler handler = new HttpClientHandler { Credentials = new NetworkCredential(key, "") })
        using (HttpClient client = new HttpClient(handler))
        {
            var request = new MultipartFormDataContent();
            request.Add(new StringContent(targetFormat), "target_format");
            request.Add(new StreamContent(File.OpenRead(sourceFile)), "source_file", new FileInfo(sourceFile).Name);
            using (HttpResponseMessage response = await client.PostAsync(url, request).ConfigureAwait(false))
            using (HttpContent content = response.Content)
            {
                string data = await content.ReadAsStringAsync();
                return JsonObject.Parse(data);
            }
        }
    }
}