import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

class DigitClassifierApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Handwritten Digit Classifier")

        self.canvas = tk.Canvas(master, width=280, height=280, bg='white')
        self.canvas.pack(padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.button_classify = tk.Button(master, text="Classify", command=self.classify_digit)
        self.button_classify.pack(pady=10)

        self.button_clear = tk.Button(master, text="Clear", command=self.clear_canvas)
        self.button_clear.pack(pady=10)

        self.model = self.load_model('mnist_cnn_model.h5')
        self.digit_image = Image.new('L', (280, 280), color='white')
        self.draw = ImageDraw.Draw(self.digit_image)

    def load_model(self, model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            return model
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.master.quit()

    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x, y, x+10, y+10, fill='black', outline='black')
        self.draw.ellipse([x, y, x+10, y+10], fill='black', outline='black')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.digit_image = Image.new('L', (280, 280), color='white')
        self.draw = ImageDraw.Draw(self.digit_image)

    def classify_digit(self):
        if self.digit_image:
            # Resize the image to 28x28 pixels and invert colors
            image = self.digit_image.resize((28, 28)).convert('L')
            image = ImageOps.invert(image)
            image = np.array(image) / 255.0
            image = image.reshape(1, 28, 28, 1)

            # Predict the digit using the loaded model
            predictions = self.model.predict(image)
            predicted_class = np.argmax(predictions[0])

            messagebox.showinfo("Prediction", f"The digit is: {predicted_class}")
        else:
            messagebox.showwarning("Warning", "Please draw a digit before classifying.")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitClassifierApp(root)
    root.mainloop()
