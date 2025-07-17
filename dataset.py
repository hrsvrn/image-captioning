import kagglehub

# Download latest version
path = kagglehub.dataset_download("aladdinpersson/flickr8kimagescaptions")

print("Path to dataset files:", path)
