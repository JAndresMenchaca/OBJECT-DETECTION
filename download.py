import urllib.request

# URL de los archivos a descargar
url_cfg = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
url_weights = "https://pjreddie.com/media/files/yolov3.weights"
url_names = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

# Guardar los archivos con sus extensiones
urllib.request.urlretrieve(url_cfg, "yolov3.cfg")
urllib.request.urlretrieve(url_weights, "yolov3.weights")
urllib.request.urlretrieve(url_names, "coco.names")

print("Archivos descargados y guardados.")
