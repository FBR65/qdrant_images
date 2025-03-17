from PIL import Image
import os

root_dir = "dataset"
output_dir = "processed/"

# Erstelle das Ausgabeverzeichnis, falls es nicht existiert
os.makedirs(output_dir, exist_ok=True)

for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        # Suchen Sie nach Bildern mit der .jpg Erweiterung
        if file.endswith(".jpg"):
            image_path = os.path.join(subdir, file)

            # Öffnen Sie das Bild
            image = Image.open(image_path)
            
            # Konvertieren Sie das Bild in ein RGBA-Format
            image = image.convert("RGBA")
            
            # Erstellen Sie ein neues Bild mit transparentem Hintergrund
            new_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
            
            # Iterieren Sie über die Pixel des Bildes
            for x in range(image.width):
                for y in range(image.height):
                    # Holen Sie sich die Pixel-Werte
                    r, g, b, a = image.getpixel((x, y))
                    
                    # Wenn der Pixel weiß ist (r=g=b=255), setzen Sie ihn transparent
                    if r == 255 and g == 255 and b == 255:
                        new_image.putpixel((x, y), (0, 0, 0, 0))
                    # Ansonsten kopieren Sie den Pixel in das neue Bild
                    else:
                        new_image.putpixel((x, y), (r, g, b, a))
            
            # Speichern Sie das neue Bild im PNG-Format, um Transparenz beizubehalten
            output_file = os.path.join(output_dir, file.replace('.jpg', '.png'))
            new_image.save(output_file)

# Entfernen Sie unerwünschte checkpoint.png Dateien aus dem Ausgabeverzeichnis
for file in os.listdir(output_dir):
    if "checkpoint.png" in file:
        os.remove(os.path.join(output_dir, file))