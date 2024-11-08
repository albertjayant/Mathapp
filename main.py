import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid GUI issues in Flask
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.integrate import simpson

app = Flask(__name__)
UPLOAD_FOLDER = 'static/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None

    curve = max(contours, key=cv2.contourArea).squeeze()
    if curve.ndim < 2:
        return None, None

    x_coords = curve[:, 0]
    y_coords = -curve[:, 1]  # Invert the y-axis to match Cartesian coordinates
    sorted_indices = np.argsort(x_coords)
    x_coords = x_coords[sorted_indices]
    y_coords = y_coords[sorted_indices]

    return x_coords, y_coords

def fit_polynomial(x, y, degree=3):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(x[:, np.newaxis])
    model = LinearRegression().fit(X_poly, y)
    coeffs = model.coef_
    intercept = model.intercept_

    equation = f"{intercept:.4f}"
    for i in range(1, len(coeffs)):
        equation += f" + {coeffs[i]:.4f} * x^{i}"

    y_fit = model.predict(X_poly)
    return y_fit, equation

def calculate_area(x, y):
    # Calculate area under the curve using Simpson's rule
    area = simpson(y, x=x)
    return abs(area)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'curve_image' not in request.files:
            return redirect(request.url)
        file = request.files['curve_image']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'curve_image.png')
            file.save(file_path)

            x, y = process_image(file_path)
            if x is None or y is None:
                return render_template('index.html', error="Curve extraction failed. Please upload a clear image.")

            y_fit, equation = fit_polynomial(x, y)
            area = calculate_area(x, y_fit)

            # Plot the original and fitted curves
            plt.figure(figsize=(8, 5))
            plt.plot(x, y, 'bo', label='Original Curve Points')
            plt.plot(x, y_fit, 'r-', label='Fitted Curve')
            plt.title('Curve Fitting and Area Calculation')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plot_path = os.path.join(app.config['UPLOAD_FOLDER'], 'plot.png')
            plt.savefig(plot_path)
            plt.close()

            return render_template('index.html', equation=equation, area=area, plot_url=plot_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
