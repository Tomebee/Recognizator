FROM mcr.microsoft.com/dotnet/core/aspnet:3.0-buster-slim AS base
WORKDIR /app
EXPOSE 80
EXPOSE 443

FROM mcr.microsoft.com/dotnet/core/sdk:3.0-buster AS build
WORKDIR /src
COPY ["Recognizator/Recognizator.csproj", "Recognizator/"]
RUN dotnet restore "Recognizator/Recognizator.csproj"
COPY . .
WORKDIR "/src/Recognizator"
RUN dotnet build "Recognizator.csproj" -c Release -o /app

FROM build AS publish
RUN dotnet publish "Recognizator.csproj" -c Release -o /app

FROM base AS final
WORKDIR /app
COPY --from=publish /app .
ENTRYPOINT ["dotnet", "Recognizator.dll"]