from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = 'uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def estimate_oxygen_delivery(leaf_area):
    oxygen_per_pixel = 0.01  # mL of oxygen per pixel (example)
    oxygen_production = leaf_area * oxygen_per_pixel
    return oxygen_production

def process_leaf_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, "Error: Could not load image."

    img = cv2.resize(image, (300, 300))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    leaf_area = cv2.countNonZero(mask)
    oxygen_production = estimate_oxygen_delivery(leaf_area)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0, 0].set_title("Original Image")
    ax[0, 1].imshow(mask, cmap='gray')
    ax[0, 1].set_title("Leaf Mask")

    v = hsv[:, :, 2]
    ax[1, 0].imshow(v, cmap='gray')
    ax[1, 0].set_title("V Channel")

    histogram, bin_edges = np.histogram(v[mask != 0], bins=256)
    ax[1, 1].plot(bin_edges[0:-1], histogram)
    ax[1, 1].set_title("Grayscale Histogram")
    
    plot_filename = os.path.join(UPLOAD_FOLDER, "leaf_analysis.png")
    plt.tight_layout()
    plt.savefig(plot_filename)

    return oxygen_production, plot_filename

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            oxygen_production, plot_filepath = process_leaf_image(filepath)
            if oxygen_production is None:
                return "Error processing the image."
            
            return render_template("result.html", oxygen=oxygen_production, plot_image=plot_filepath)

    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)
