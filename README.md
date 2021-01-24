# Recognizator
A ML .NET project that classifies images from 3 categories
- Cat
- Dog
- Cookie

Presently there are three model training methods to choose from:
- Transfer learning with Inception
- Gpu supported Inception retrain
- Gpu supported Resnet retrain

## Prerequisites
- .NET Core 3.0

## Getting started
1. Recognizator uses **Nvidia CUDA**, therefore follow instructions from https://github.com/dotnet/machinelearning/blob/master/docs/api-reference/tensorflow-usage.md

2. Download inception model from https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip , and copy its content to `Recognizator/AI/Training/Strategies/Inception/Model/`

3. Run project from vs code using
`dotnet run --project Recognizator/Recognizator.csproj`
or debug in code/vs

4. Import two jsons in root folder to your postman and enjoy

## Cats and dogs images
In case of more prediction score needed(dogs and/or cats):
[Cats and dogs dataset](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs)

## Roadmap
[] Docker support with tensorflow preinstalled and CUDA

[] Deep learning from scratch as another training strategy

[] One click run (includes automated files downloading and so on)
