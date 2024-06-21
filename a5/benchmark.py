from sys import argv
from resizeable_image import ResizeableImage

filename = argv[1]

dp = argv[2] != "naive"

image = ResizeableImage(filename)
while image.width > 0:
    seam = image.best_seam(dp=dp)
    image.remove_best_seam()
