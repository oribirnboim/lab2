import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import matplotlib.pyplot as plt


class PhotoAnnotator:
    def __init__(self, root, folder_path):
        self.root = root
        self.folder_path = folder_path
        self.images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
        self.current_image_index = 0
        self.image_data = []
        self.lines = []

        self.root.bind('<z>', self.undo_last_line)
        self.root.bind('<Return>', self.next_image)

        self.canvas = tk.Canvas(root)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<Button-1>', self.start_line)
        self.canvas.bind('<B1-Motion>', self.draw_line)
        self.canvas.bind('<ButtonRelease-1>', self.stop_line)

        self.load_image()

    def load_image(self):
        if self.current_image_index >= len(self.images):
            self.prompt_save_json()
            self.show_histograms()
            return

        image_path = os.path.join(self.folder_path, self.images[self.current_image_index])
        self.image = Image.open(image_path)
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.config(width=self.tk_image.width(), height=self.tk_image.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.lines = []
        self.redraw()

    def start_line(self, event):
        self.start_x, self.start_y = event.x, event.y
        self.current_line = [(event.x, event.y)]

    def draw_line(self, event):
        self.canvas.create_line(self.current_line[-1][0], self.current_line[-1][1], event.x, event.y, fill='red')
        self.current_line.append((event.x, event.y))

    def stop_line(self, event):
        if hasattr(self, 'current_line') and len(self.current_line) > 1:
            self.lines.append(self.current_line)
            del self.current_line

    def undo_last_line(self, event):
        if self.lines:
            self.lines.pop()
            self.redraw()

    def next_image(self, event):
        self.save_image_data()
        self.current_image_index += 1
        self.load_image()

    def redraw(self):
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        for line in self.lines:
            for i in range(1, len(line)):
                self.canvas.create_line(line[i - 1][0], line[i - 1][1], line[i][0], line[i][1], fill='red')

    def save_image_data(self):
        line_lengths = [sum(((line[i][0] - line[i - 1][0]) ** 2 + (line[i][1] - line[i - 1][1]) ** 2) ** 0.5 for i in
                            range(1, len(line))) for line in self.lines]
        self.image_data.append({
            'image_name': self.images[self.current_image_index],
            'line_lengths': line_lengths,
            'number_of_lines': len(self.lines)
        })

    def prompt_save_json(self):
        save_path = filedialog.asksaveasfilename(
            defaultextension='.json',
            filetypes=[('JSON files', '*.json')],
            title='Save Image Data',
            initialdir=self.folder_path
        )
        if save_path:
            self.save_data_to_json(save_path)
        else:
            messagebox.showinfo('Info', 'Save operation cancelled.')

    def save_data_to_json(self, save_path):
        with open(save_path, 'w') as json_file:
            json.dump(self.image_data, json_file, indent=4)
        messagebox.showinfo('Info', f'Image data saved to {save_path}')

    def show_histograms(self):
        if not self.image_data:
            messagebox.showinfo('Info', 'No lines drawn.')
            self.root.quit()
            return

        self.plot_histogram_line_lengths()
        self.plot_histogram_number_of_lines()
        self.root.quit()

    def plot_histogram_line_lengths(self):
        line_lengths = [length for image in self.image_data for length in image['line_lengths']]
        if line_lengths:
            plt.hist(line_lengths, bins=20)
            plt.xlabel('Line Length')
            plt.ylabel('Frequency')
            plt.title('Histogram of Line Lengths')
            plt.show()

    def plot_histogram_number_of_lines(self):
        number_of_lines = [image['number_of_lines'] for image in self.image_data]
        if number_of_lines:
            plt.hist(number_of_lines, bins=40)
            plt.xlabel('Number of Jumps')
            plt.ylabel('Frequency')
            plt.title('Histogram of Number of Lines per Image')
            plt.show()


def plot_histogram_from_json(json_path):
    with open(json_path, 'r') as json_file:
        image_data = json.load(json_file)

    line_lengths = [length for image in image_data for length in image['line_lengths']]
    number_of_lines = [image['number_of_lines'] for image in image_data]

    if line_lengths:
        plt.hist(line_lengths, bins=20)
        plt.xlabel('Line Length')
        plt.ylabel('Frequency')
        plt.title('Histogram of Line Lengths from JSON')
        plt.show()

    if number_of_lines:
        plt.hist(number_of_lines, bins=20)
        plt.xlabel('Number of Jumps')
        plt.ylabel('Frequency')
        plt.title('Histogram for Number of Jumps')
        plt.show()


def main():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title='Select Folder with JPG Photos')
    if not folder_path:
        messagebox.showerror('Error', 'No folder selected')
        return

    root.deiconify()
    app = PhotoAnnotator(root, folder_path)
    root.mainloop()


if __name__ == '__main__':
    main()
    plot_histogram_from_json('p3b1_jumps.json')
    # plot_histogram_from_json('test_lengths.json')
